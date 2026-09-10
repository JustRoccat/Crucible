import argparse
import torch
from src.model import NeuroSymbolicModel
from src.backbone import build_backbone
from src.tokenizer import ProductionTokenizer
from src.hdc import HDCMemory
from src.symbolic import SymbolicSpec


def wrap_prompt(raw_prompt: str) -> str:
    return f"### Instruction\n{raw_prompt}\n\n### Response\n"


@torch.no_grad()
def generate(
    model,
    prompt_ids: list,
    n_tokens: int,
    device,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.25,
    greedy: bool = False,
):
    model.eval()
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(n_tokens):
        out = model(tokens[:, -256:])
        logits = out["lm_logits"][:, -1, :]
        if greedy:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            continue
        logits = logits / temperature
        for token_id in set(tokens[0].tolist()):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= repetition_penalty
            else:
                logits[0, token_id] /= repetition_penalty
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens[0].tolist()


def _parse_symbolic_spec(text: str):
    var_names, constraints = ([], [])
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("VARS:"):
            names_part = line.split(":", 1)[1]
            var_names = [
                n.strip() for n in names_part.split(",") if n.strip().isidentifier()
            ]
        elif line.upper().startswith("CONSTRAINT:"):
            c = line.split(":", 1)[1].strip()
            if c:
                constraints.append(c)
    return (var_names, constraints)


def build_model_propose_fn(
    model, tokenizer, device, n_tokens: int = 80, greedy: bool = True
):

    def propose_fn(context: str, history: list) -> SymbolicSpec:
        instruction = f"Express the problem below as integer constraints for a solver.\nUse ONLY: variable names, comparisons (< <= > >= == !=), and/or/not, and + - * // between integers/variables.\nOutput EXACTLY this format, nothing else:\nVARS: <comma-separated variable names>\nCONSTRAINT: <one constraint>\n(one CONSTRAINT line per constraint)\n\n{context}"
        wrapped = wrap_prompt(instruction)
        prompt_ids = tokenizer.encode(wrapped)
        generated_ids = generate(model, prompt_ids, n_tokens, device, greedy=greedy)
        full_text = tokenizer.decode(generated_ids)
        response = (
            full_text[len(wrapped) :] if full_text.startswith(wrapped) else full_text
        )
        var_names, constraints = _parse_symbolic_spec(response)
        if not var_names or not constraints:
            return SymbolicSpec(
                var_names=["_parse_failure"],
                constraints=["_parse_failure > 0", "_parse_failure < 0"],
                description=f"PARSE_ERROR: the model did not generate the VARS:/CONSTRAINT: format. Raw response (first 200 chars): {response[:200]!r}",
            )
        return SymbolicSpec(
            var_names=var_names,
            constraints=constraints,
            description=response.strip()[:300],
        )

    return propose_fn


def _demo_propose_fn(context: str, history: list) -> SymbolicSpec:
    if not history:
        return SymbolicSpec(
            var_names=["n"],
            constraints=["n > 0", "n < 10", "n < 0"],
            description="attempt 1: invalid array index specification",
        )
    return SymbolicSpec(
        var_names=["n"],
        constraints=["n > 0", "n < 10"],
        description="attempt 2: fixed after feedback from Z3",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--n_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.25)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Deterministic argmax instead of sampling (diagnostics).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Do not wrap the prompt in ### Instruction / ### Response.",
    )
    parser.add_argument(
        "--proof_demo",
        action="store_true",
        help="Show the Model<->Z3 loop mechanism on a detached, fixed example (mock).",
    )
    parser.add_argument(
        "--z3_solve",
        action="store_true",
        help="The SAN model formalizes --prompt as a SymbolicSpec and Z3 verifies it (a real bridge).",
    )
    parser.add_argument(
        "--z3_gen_tokens",
        type=int,
        default=80,
        help="How many tokens to generate on each formalization attempt in --z3_solve.",
    )
    args = parser.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cpu")
    tokenizer = ProductionTokenizer.from_config(cfg["tokenizer"])
    backbone_cfg = dict(cfg["backbone"])
    if backbone_cfg.get("mode", "custom_mamba") == "custom_mamba":
        backbone_cfg["vocab_size"] = tokenizer.vocab_size
    backbone = build_backbone(backbone_cfg).to(device)
    vocab_size_for_head = getattr(backbone, "vocab_size", tokenizer.vocab_size)
    model = NeuroSymbolicModel(
        backbone=backbone,
        vocab_size=vocab_size_for_head,
        ttt_rank=cfg["model"]["ttt_rank"],
        ttt_inner_lr=cfg["model"]["ttt_inner_lr"],
        ttt_inner_steps=cfg["model"]["ttt_inner_steps"],
        jepa_hidden_mult=cfg["model"]["jepa_hidden_mult"],
        ema_decay=cfg["model"]["ema_decay"],
        hdc_dim=cfg["model"]["hdc_dim"],
        mask_ratio=cfg["model"]["mask_ratio"],
        max_self_correction_attempts=cfg["model"]["max_self_correction_attempts"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    effective_prompt = args.prompt if args.raw else wrap_prompt(args.prompt)
    print("=== Prompt sent to the model ===")
    print(repr(effective_prompt))
    print()
    prompt_ids = tokenizer.encode(effective_prompt)
    generated = generate(
        model,
        prompt_ids,
        args.n_tokens,
        device,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        greedy=args.greedy,
    )
    text = tokenizer.decode(generated)
    print("=== Generated text ===")
    print(text)
    hdc_mem = HDCMemory(dim=cfg["model"]["hdc_dim"])
    with torch.no_grad():
        model(torch.tensor([prompt_ids], device=device), hdc_memory=hdc_mem)
    print("\n=== HDC Memory (representation read/write) ===")
    print("Stored keys:", list(hdc_mem._item_memory.keys()))
    if args.z3_solve:
        print("\n=== Model -> Z3: prompt formalization and verification ===")
        print(f"Context: {args.prompt!r}")
        propose_fn = build_model_propose_fn(
            model, tokenizer, device, n_tokens=args.z3_gen_tokens, greedy=args.greedy
        )
        outcome = model.self_correct(
            propose_fn, initial_context=args.prompt, hdc_memory=hdc_mem
        )
        for att in outcome.attempts:
            print(f"\n  Attempt {att.attempt_idx}:")
            print(f"    VARS={att.spec.var_names}  CONSTRAINTS={att.spec.constraints}")
            if att.spec.description.startswith("PARSE_ERROR"):
                print(f"    {att.spec.description}")
            print(f"    -> {att.result.status}: {att.result.message}")
        print(f"\nSuccess: {outcome.success} after {outcome.n_attempts} attempts.")
        print(f"Patterns verified in HDC Memory: {hdc_mem.n_verified_patterns()}")
    elif args.proof_demo:
        print("\n=== Model<->Z3 self-correction loop (mock, detached from --prompt) ===")
        outcome = model.self_correct(
            _demo_propose_fn,
            initial_context="Verify the array index.",
            hdc_memory=hdc_mem,
        )
        for att in outcome.attempts:
            print(
                f"  Attempt {att.attempt_idx}: {att.result.status} — {att.result.message}"
            )
        print(f"Success: {outcome.success} after {outcome.n_attempts} attempts.")
        print(f"Patterns verified in HDC Memory: {hdc_mem.n_verified_patterns()}")


if __name__ == "__main__":
    main()
