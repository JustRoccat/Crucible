import os
import re
import random
from datasets import load_dataset

TARGET_TOTAL_MB = 2000
OUTPUT_PATH = "corpus.txt"
EOS_MARKER = "<|endoftext|>"
RANDOM_SEED = 42
MAX_SAMPLES_PER_DATASET = 300000
USE_LARGE_STACK = False
random.seed(RANDOM_SEED)
BYTES_TOTAL = TARGET_TOTAL_MB * 1024 * 1024
CATEGORY_WEIGHTS = {
    "1_instructions": 0.2,
    "2_reasoning_cot": 0.15,
    "3_multi_file": 0.2,
    "4_bugfix_verify": 0.2,
    "5_multi_lang_code": 0.25,
}
assert abs(sum(CATEGORY_WEIGHTS.values()) - 1.0) < 1e-06


def fmt(**fields):
    parts = []
    for header, content in fields.items():
        if not content:
            continue
        clean_header = header.replace("_", " ")
        parts.append(f"### {clean_header}\n{content}".rstrip())
    if not parts:
        return None
    return "\n\n".join(parts)


def extract_kodcode(row):
    q = row.get("question") or ""
    sol = row.get("solution") or row.get("gpt_response") or ""
    test = row.get("test") or row.get("test_info") or ""
    if not q or not sol:
        return None
    return fmt(Instruction=q, Tests=test, Response=sol)


def extract_glaive(row):
    q = row.get("question") or row.get("input") or ""
    a = row.get("answer") or row.get("output") or ""
    if not q or not a:
        return None
    return fmt(Instruction=q, Response=a)


def extract_opencodeinstruct(row):
    q = row.get("input") or row.get("instruction") or row.get("question") or ""
    a = row.get("output") or row.get("solution") or ""
    if not q or not a:
        return None
    return fmt(Instruction=q, Response=a)


def extract_mixture_of_thoughts_code(row):
    msgs = row.get("messages")
    if not msgs:
        return None
    user_parts = [m["content"] for m in msgs if m.get("role") == "user"]
    asst_parts = [m["content"] for m in msgs if m.get("role") == "assistant"]
    instruction = "\n".join(user_parts)
    response = "\n".join(asst_parts)
    if not instruction or not response:
        return None
    return fmt(Instruction=instruction, Reasoning_and_Response=response)


def extract_openr1_math(row):
    problem = row.get("problem") or ""
    gens = row.get("generations") or []
    trace = gens[0] if gens else row.get("solution", "")
    if not problem or not trace:
        return None
    return fmt(Instruction=problem, Reasoning=trace)


def extract_codeforces_cots(row):
    problem = row.get("prompt") or row.get("problem") or ""
    sol = row.get("generation") or row.get("response") or ""
    if not problem or not sol:
        return None
    return fmt(Instruction=problem, Reasoning_and_Solution=sol)


def extract_swebench(row):
    issue = row.get("problem_statement") or ""
    patch = row.get("patch") or ""
    repo = row.get("repo") or ""
    if not issue or not patch:
        return None
    context = f"Repository: {repo}\n\n{issue}" if repo else issue
    return fmt(Instruction=context, Response=patch)


def extract_swebench_bm25(row):
    text = row.get("text") or ""
    patch = row.get("patch") or ""
    if not text or not patch:
        return None
    return fmt(Instruction=text, Response=patch)


def extract_commitpackft(row):
    msg = row.get("subject") or row.get("commit_message") or ""
    old = row.get("old_contents") or ""
    new = row.get("new_contents") or ""
    if not new:
        return None
    instruction = f"{msg}\n\nOriginal code:\n{old}" if old else msg
    return fmt(Instruction=instruction, Response=new)


def extract_editpackft(row):
    instr = row.get("instruction") or row.get("subject") or row.get("message") or ""
    old = row.get("old_contents") or row.get("before") or ""
    new = row.get("new_contents") or row.get("after") or ""
    if not new or not instr:
        return None
    instruction = f"{instr}\n\nOriginal code:\n{old}" if old else instr
    return fmt(Instruction=instruction, Response=new)


def extract_stack(row):
    content = row.get("content") or ""
    lang = row.get("lang") or ""
    if not content:
        return None
    return fmt(Language=lang, Code=content)


STACK_REPO = "bigcode/the-stack-dedup" if USE_LARGE_STACK else "bigcode/the-stack-smol"
STACK_LANGUAGES = [
    ("data/python", 3),
    ("data/c", 2),
    ("data/c++", 2),
    ("data/rust", 2),
    ("data/go", 1),
    ("data/shell", 1),
    ("data/sql", 1),
]
DATASETS = {
    "1_instructions": [
        dict(
            repo="KodCode/KodCode-V1-SFT-R1",
            config=None,
            split="train",
            extractor=extract_kodcode,
            weight=3,
        ),
        dict(
            repo="nvidia/OpenCodeInstruct",
            config=None,
            split="train",
            extractor=extract_opencodeinstruct,
            weight=3,
        ),
        dict(
            repo="glaiveai/glaive-code-assistant",
            config=None,
            split="train",
            extractor=extract_glaive,
            weight=1,
        ),
    ],
    "2_reasoning_cot": [
        dict(
            repo="open-r1/Mixture-of-Thoughts",
            config="code",
            split="train",
            extractor=extract_mixture_of_thoughts_code,
            weight=4,
        ),
        dict(
            repo="open-r1/codeforces-cots",
            config="solutions",
            split="train",
            extractor=extract_codeforces_cots,
            weight=4,
        ),
        dict(
            repo="open-r1/OpenR1-Math-220k",
            config="default",
            split="train",
            extractor=extract_openr1_math,
            weight=1,
        ),
    ],
    "3_multi_file": [
        dict(
            repo="SWE-bench/SWE-bench",
            config=None,
            split="test",
            extractor=extract_swebench,
            weight=1,
        ),
        dict(
            repo="princeton-nlp/SWE-bench_Verified",
            config=None,
            split="test",
            extractor=extract_swebench,
            weight=1,
        ),
        dict(
            repo="princeton-nlp/SWE-bench_bm25_27K",
            config=None,
            split="train",
            extractor=extract_swebench_bm25,
            weight=8,
        ),
    ],
    "4_bugfix_verify": [
        *[
            dict(
                repo="bigcode/commitpackft",
                config=None,
                data_dir=lang,
                split="test",
                extractor=extract_commitpackft,
                weight=w,
                revision="refs/convert/parquet",
            )
            for lang, w in [
                ("python", 4),
                ("javascript", 3),
                ("typescript", 2),
                ("java", 2),
                ("go", 2),
                ("rust", 2),
                ("c++", 2),
                ("c", 1),
                ("shell", 1),
                ("sql", 1),
            ]
        ],
        dict(
            repo="nuprl/EditPackFT-Multi",
            config=None,
            split="train",
            extractor=extract_editpackft,
            weight=3,
        ),
    ],
    "5_multi_lang_code": [
        dict(
            repo=STACK_REPO,
            config=data_dir,
            split="train",
            extractor=extract_stack,
            weight=weight,
        )
        for data_dir, weight in STACK_LANGUAGES
    ],
}


def _open_source(e):
    repo_id, config, split, extractor, weight = (
        e["repo"],
        e["config"],
        e["split"],
        e["extractor"],
        e["weight"],
    )
    revision = e.get("revision")
    data_dir = e.get("data_dir")
    if data_dir is None and config and str(config).startswith("data/"):
        data_dir = config
        config = None

    def _try_load(rev, sp):
        kwargs = dict(split=sp, streaming=True)
        if rev:
            kwargs["revision"] = rev
        if data_dir:
            return load_dataset(repo_id, data_dir=data_dir, **kwargs)
        elif config:
            return load_dataset(repo_id, config, **kwargs)
        else:
            return load_dataset(repo_id, **kwargs)

    try:
        ds = _try_load(revision, split)
    except Exception as ex:
        handled = False
        m = re.search("Bad split:.*Available splits:\\s*\\[([^\\]]*)\\]", str(ex))
        if m:
            available = [
                s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()
            ]
            if available:
                fallback_split = available[0]
                print(
                    f"  [INFO] {repo_id}: split '{split}' unavailable, trying '{fallback_split}' (available: {available})..."
                )
                try:
                    ds = _try_load(revision, fallback_split)
                    split = fallback_split
                    handled = True
                except Exception as ex2:
                    ex = ex2
        if (
            not handled
            and revision is None
            and ("Dataset scripts are no longer supported" in str(ex))
        ):
            try:
                print(
                    f"  [INFO] {repo_id}: loading script rejected, trying refs/convert/parquet branch..."
                )
                ds = _try_load("refs/convert/parquet", split)
                handled = True
            except Exception as ex2:
                ex = ex2
        if not handled:
            print(
                f"  [SKIPPED] {repo_id} ({config or data_dir or 'default'}) -> load error: {ex}"
            )
            return None
    return dict(
        repo_id=repo_id,
        label=config or data_dir or "default",
        extractor=extractor,
        weight=weight,
        iterator=iter(ds),
        written=0,
        n_samples=0,
        exhausted=False,
        blocks=[],
    )


def _consume(state, extra_budget):
    if (
        extra_budget <= 0
        or state["exhausted"]
        or state["n_samples"] >= MAX_SAMPLES_PER_DATASET
    ):
        return 0
    target = state["written"] + extra_budget
    got = 0
    try:
        for row in state["iterator"]:
            if (
                state["written"] >= target
                or state["n_samples"] >= MAX_SAMPLES_PER_DATASET
            ):
                return got
            try:
                block = state["extractor"](row)
            except Exception:
                block = None
            if not block:
                continue
            state["blocks"].append(block)
            nbytes = len(block.encode("utf-8"))
            state["written"] += nbytes
            state["n_samples"] += 1
            got += nbytes
        state["exhausted"] = True
    except Exception as ex:
        print(f"     [STREAM INTERRUPTED] {state['repo_id']} ({state['label']}): {ex}")
        state["exhausted"] = True
    return got


def collect_category(category_name, entries, byte_budget, max_rounds=4):
    print(
        f"\n=== Category: {category_name} | budget: {byte_budget / 1000000.0:.1f} MB ==="
    )
    states = [s for s in (_open_source(e) for e in entries) if s is not None]
    if not states:
        return ([], 0, [])
    remaining = byte_budget
    for _ in range(max_rounds):
        active = [
            s
            for s in states
            if not s["exhausted"] and s["n_samples"] < MAX_SAMPLES_PER_DATASET
        ]
        if not active or remaining <= 0:
            break
        total_w = sum((s["weight"] for s in active))
        progress = 0
        for s in active:
            share = int(remaining * (s["weight"] / total_w)) if total_w else 0
            progress += _consume(s, share)
        remaining = byte_budget - sum((s["written"] for s in states))
        if progress == 0:
            break
    all_blocks = []
    for s in states:
        if (
            s["written"]
            < byte_budget * (s["weight"] / sum((x["weight"] for x in states))) * 0.5
        ):
            print(
                f"     [WARNING] {s['repo_id']} ({s['label']}) ran out significantly below its budget share ({s['written'] / 1000000.0:.1f} MB)."
            )
        print(
            f"     {s['repo_id']} ({s['label']}): collected {s['n_samples']} samples, {s['written'] / 1000000.0:.2f} MB{(' [exhausted]' if s['exhausted'] else '')}"
        )
        all_blocks.extend(s["blocks"])
    total_written = sum((s["written"] for s in states))
    print(
        f"  TOTAL category {category_name}: {total_written / 1000000.0:.2f} MB (budget {byte_budget / 1000000.0:.1f} MB)"
    )
    return (all_blocks, total_written, states)


def main():
    print(f"Target corpus size: {TARGET_TOTAL_MB} MB -> {OUTPUT_PATH}")
    print(
        f"Code source (category 5): {STACK_REPO} ({('FULL dataset, requires HF login' if USE_LARGE_STACK else 'small smol demo subset')})"
    )
    print(f"Document boundary separator: {EOS_MARKER!r}")
    all_blocks = []
    all_states = []
    total_written = 0
    for cat_name, weight in CATEGORY_WEIGHTS.items():
        byte_budget = int(BYTES_TOTAL * weight)
        blocks, written, states = collect_category(
            cat_name, DATASETS[cat_name], byte_budget
        )
        all_blocks.extend(blocks)
        all_states.extend(states)
        total_written += written
    shortfall = BYTES_TOTAL - total_written
    if shortfall > BYTES_TOTAL * 0.02:
        print(
            f"\n=== Redistributing across categories: short by {shortfall / 1000000.0:.1f} MB, topping up sources with spare capacity ==="
        )
        for _ in range(4):
            growable = [
                s
                for s in all_states
                if not s["exhausted"] and s["n_samples"] < MAX_SAMPLES_PER_DATASET
            ]
            if not growable or shortfall <= 0:
                break
            total_w = sum((s["weight"] for s in growable))
            progress = 0
            for s in growable:
                share = int(shortfall * (s["weight"] / total_w)) if total_w else 0
                got = _consume(s, share)
                progress += got
                if got:
                    print(
                        f"     +{got / 1000000.0:.2f} MB from {s['repo_id']} ({s['label']}) [total now {s['written'] / 1000000.0:.2f} MB]"
                    )
            shortfall = BYTES_TOTAL - sum((s["written"] for s in all_states))
            if progress == 0:
                break
        all_blocks = [b for s in all_states for b in s["blocks"]]
        total_written = sum((s["written"] for s in all_states))
    random.shuffle(all_blocks)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for block in all_blocks:
            out_f.write(block)
            out_f.write(f"\n\n{EOS_MARKER}\n\n")
    final_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(
        f"\nDONE. {len(all_blocks)} examples, final size of {OUTPUT_PATH}: {final_size:.2f} MB"
    )
    if final_size < TARGET_TOTAL_MB * 0.7:
        print(
            f"[WARNING] The result ({final_size:.0f} MB) is noticeably below the target ({TARGET_TOTAL_MB} MB) - all sources in all categories were exhausted down to the bottom (see [exhausted] above). The only way to go further is to add NEW data sources to DATASETS, most likely in category 5 (log in and set USE_LARGE_STACK=True)."
        )
    print(
        "Note: this file is NOT tokenized or split into windows by this script - that is done by TokenizedTextDataset when train.py starts."
    )


if __name__ == "__main__":
    main()
