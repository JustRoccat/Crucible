# Training Mamba/Mamba2 on Kaggle (dual T4) — what works, what to avoid

Document written based on real runs of training the SAN 2.0 adapters
(JEPA + TTT + HDC + Z3) on a frozen Mamba2 2.7B backbone, on Kaggle
Notebooks with a T4 ×2 GPU accelerator. Each section covers a concrete
problem we ran into, and the solution that worked.

---

## 1. Choosing a Mamba/Mamba2 model from the Hugging Face Hub

**Problem:** The official `state-spaces` organization on the HF Hub **does not publish**
Mamba2 checkpoints in a format compatible with `transformers.AutoModel`.
`state-spaces/mamba2-2.7b` (without `-hf`) is a raw checkpoint for the
`mamba-ssm` library, not loadable via `AutoModel.from_pretrained()`.

**What works:**
- For **Mamba1**: official, compatible repos exist —
  `state-spaces/mamba-130m-hf`, `mamba-1.4b-hf`, `mamba-2.8b-hf`.
- For **Mamba2**: you need to use a community conversion, e.g.
  `AntonV/mamba2-2.7b-hf` (the same weights, repackaged into the `-hf` format).
  It's not an official `state-spaces` repo, but the weights are identical.

**Important conceptual distinction:** `transformers` is the name of a
Hugging Face *library*, not an architecture. The library supports dozens
of architectures, including pure Mamba (SSM, zero attention). The `-hf`
suffix in a repo name just means "this format loads with AutoModel", not
"this is a transformer".

**Going forward:** before starting, check on the HF Hub whether the given
model size/version has an `-hf` variant at all (or a community conversion)
before you plan your whole pipeline around it.

---

## 2. CUDA kernels (`causal-conv1d`, `mamba-ssm`) — installation and whether they're worth it

**Problem:** The first install attempt (`pip install causal-conv1d>=1.4.0`)
failed without a readable error (because `-q`/quiet was used). Installing
without `-q` revealed the real compilation error the first time, but **on
the next attempt, in a fresh session, it built correctly** — the Kaggle
image environment can be inconsistent between sessions.

**What works:**
```bash
pip install --break-system-packages causal-conv1d --no-build-isolation
pip install --break-system-packages mamba-ssm --no-build-isolation
```
No `-q`, so you can see the real error if it fails. `--no-build-isolation`
helps avoid `torch` version mismatches between the build environment and
the target environment.

**Surprising finding — the kernels don't actually help with training anyway:**
The official `transformers` documentation for Mamba2 states outright: *without
compilation (torch.compile), `torch_forward` (the "slow", pure PyTorch
path) is 3-4× faster than `cuda_kernels_forward` on long sequences
("prefill")*. `cuda_kernels_forward` is optimized for token-by-token
generation with cache (inference), not for training on full sequences.
`transformers` **deliberately** chooses `torch_forward` during training,
regardless of whether the CUDA kernels are installed.

**Going forward:** if you're training (not generating with cache), don't
waste time installing `causal-conv1d`/`mamba-ssm` hoping for a speedup
— it won't work the way it does for most other HF models. Only install
them if you actually plan to run `model.generate()` with cache after training.

---

## 3. OutOfMemoryError on Mamba2's `torch_forward` — the real cause

**Problem:** `CUDA out of memory. Tried to allocate 40.00 GiB` on the
first forward pass, even though the (frozen) backbone only takes up
~5.4GB, and the card has 16GB.

**Cause:** `torch_forward` (the path used during training, see section 2)
computes in fp32 and creates an intermediate tensor with size proportional
to `batch_size × seq_len × chunk_size × num_heads × state_size`. The
default `chunk_size` (256), combined with a reasonable batch/seq_len,
explodes to tens of GB.

**Memory formula (approximate):**
```
memory[bytes] ≈ batch_size × seq_len × chunk_size × num_heads × state_size × 4
```
(linear in each of these parameters — reducing `chunk_size` by 8× gives
8× less memory for this particular tensor).

**What works:** overriding `chunk_size` in the model config (the
`Mamba2Config.chunk_size` field, 256 by default) to a much smaller value,
e.g. **32**. This has to be done at the `AutoConfig` level, because
`AutoModel.from_pretrained` doesn't safely accept it as a plain `kwarg`
across all `transformers` versions:
```python
config = AutoConfig.from_pretrained(repo_id)
config.chunk_size = 32
model = AutoModel.from_pretrained(repo_id, config=config, ...)
```
Also helps: `os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"`
(reduces CUDA memory fragmentation — in our logs we saw close to 5GB
"reserved but unused" by PyTorch).

**Going forward:** on 16GB cards with a model >1B parameters, always
start with a very small `chunk_size` (32), a small `batch_size` (1-2),
and a small `seq_len` (256), verify it works, and only then carefully
scale up one at a time, measuring VRAM usage (`nvidia-smi`) after each
change.

---

## 4. `torch_dtype` on T4 (Turing, sm_75)

**Problem:** the default `bfloat16` in many example configs/scripts
(copied from models trained for Ampere+) doesn't make sense on T4.

**Cause:** T4 is a Turing architecture card (compute capability sm_75).
Hardware acceleration for `bfloat16` on tensor cores only exists from
Ampere (sm_80) onward. On T4, `bfloat16` "works", but doesn't get the
tensor-core speedup that `float16` does.

**What works:** detect compute capability at runtime and force `float16`
below sm_80:
```python
major, _ = torch.cuda.get_device_capability(0)
dtype = torch.bfloat16 if major >= 8 else torch.float16
```

**Going forward:** never copy `torch_dtype: bfloat16` from a config
written for A100/H100 without checking the target card's architecture.

---

## 5. `pip install` on Kaggle vs. locally — `--break-system-packages`

**Problem:** the same `pip install ... --break-system-packages` command
runs fine on Kaggle, but throws an `externally-managed-environment` error
on Linux distributions with PEP 668 (e.g. Gentoo, newer Debian/Ubuntu).

**What works:**
- **On Kaggle** (an ephemeral container that disappears after the session):
  `--break-system-packages` is fine, there's nothing permanent to "break".
- **Locally**: always use a `venv`, never `--break-system-packages` on the
  system Python:
  ```bash
  python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
  ```

**Going forward:** clearly separate, in docs/scripts, commands meant
"to paste into Kaggle" from commands "to run locally".

---

## 6. Jupyter/Kaggle: `!python -c "multiline string"` doesn't work

**Problem:** repeatedly hit `SyntaxError: unterminated string literal`
when trying to paste a multiline Python script as `!python -c "..."`
in a Jupyter/Kaggle cell.

**Cause:** `!command` in Jupyter passes each line to the shell
separately — multiline quotes don't join up the way they do in a normal
bash terminal.

**What works:** always use a separate file via `%%writefile` (must be the
first line of the cell), then `!python file.py` in the next cell:
```python
%%writefile script.py
# ... multiline code ...
```
```python
!python script.py
```

**Going forward:** for any code longer than one line, use `%%writefile`
straight away, not `!python -c`.

---

## 7. `subprocess.call(cmd, shell=True)` with unquoted special characters

**Problem:** the command `pip install causal-conv1d>=1.4.0` run via
`subprocess.call(f"... {pkg}", shell=True)` without quotes accidentally
created files `=1.4.0`/`=2.2.0` in the working directory.

**Cause:** the bash shell interpreted `>=1.4.0` as a redirect (`>`) into
a file called `=1.4.0`, because `>=` wasn't quoted.

**Going forward:** always quote version specifiers in commands built with
`shell=True`: `f'pip install "{pkg}"'`, or use
`subprocess.run([...], shell=False)` with a list of arguments instead of
a string.

---

## 8. Saving checkpoints — don't save the frozen backbone

**Problem:** `torch.save({"model": model.state_dict(), ...})` saves
**all** parameters, including the frozen 2.7B backbone (~5.4GB in fp16)
— even though the backbone doesn't change and gets reconstructed from
the HF Hub on every script start. Effect: filled up the 20GB
`/kaggle/working` disk quota after just 2-3 checkpoints, plus a write
error mid-file (`RuntimeError: unexpected pos...`) when the disk ran out
during a write.

**What works:**
```python
def trainable_state_dict(model):
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in trainable}
```
When loading: `model.load_state_dict(ckpt["model"], strict=False)` —
missing keys (the whole frozen backbone) are expected and safe to
ignore, because the backbone is rebuilt from scratch from the HF Hub
before the checkpoint is loaded.

Additionally: delete the previous `step_N.pt` when saving a new one, so
you don't multiply files unnecessarily — keep only the most recent
periodic checkpoint plus an optional `final.pt`.

**Going forward:** for any training run with a frozen large backbone,
design checkpointing from the start to save **only the trainable
parameters**. This also speeds up `torch.save`/`push_to_hub` between
sessions.

---

## 9. `/kaggle/working` disk limit (20GB) — watch the model cache

**Problem:** while experimenting with different backbone repos (first
the wrong `state-spaces/mamba2-2.7b-hf`, then a fallback to
`mamba-1.4b-hf`, finally the correct `AntonV/mamba2-2.7b-hf`),
`backbone_cache/` ended up with **two** full models (~11GB), on top of
the training checkpoints.

**What works:**
```bash
!du -sh /kaggle/working/* /kaggle/working/backbone_cache/*   # see what's taking up space
!rm -rf /kaggle/working/backbone_cache/models--<unneeded-model>
```

**Going forward:** after every change of the backbone's `repo_id`, check
and clean up `backbone_cache/` before moving on. It's worth checking
`df -h /kaggle/working` regularly, not only once something blows up.

---

## 10. The GPU accelerator eats into the 30h/week limit for the **whole session**, not
just during computation

**Key understanding:** Kaggle's 30h GPU/week limit is counted from the
moment a session starts with a GPU accelerator assigned, **regardless**
of whether GPU computation is actually happening at that moment (`ls`,
`pip install`, editing a config — all of this "costs" in the background
if the accelerator is set to GPU).

**What works:**
- **Accelerator: None/CPU** for prep work (browsing files, writing
  configs, checking the dataset structure).
- **Accelerator: GPU T4 ×2** only once you're actually running something
  that needs a GPU (`download_backbone.py`, `train.py`).
- **Always explicitly stop the session** (the power button ⏻) once
  you're done working — closing the browser tab does NOT end the session
  automatically and can silently eat into the limit in the background.

---

## 11. Notebook autosave can be unreliable — "Failed to save draft"

**Problem:** the notebook repeatedly showed `Failed to save draft` at the
top of the screen, which resulted in **losing all cells** except the
first one after a refresh/session restart.

**What works:**
- Click **`Save Version`** (top right corner) regularly, after adding a
  few cells — this is an explicit, durable notebook checkpoint,
  independent of the unreliable autosave.
- Consolidate related commands into fewer, larger cells instead of many
  small ones — fewer chances for an interrupted save.

---

## 12. Checklist for starting a new session (after a restart/new limit week)

1. `Accelerator: GPU T4 ×2`, `Internet: On` (Settings).
2. Copy the project from `/kaggle/input/.../` to the writable
   `/kaggle/working/proj/` (`/kaggle/input` is read-only).
3. Log in to the HF Hub via `UserSecretsClient().get_secret("HF_TOKEN")`
   — the token needs to be logged in again in every session.
4. `pip install -q -U datasets huggingface_hub --break-system-packages`
   (torch/transformers are already preinstalled on the Kaggle image).
5. Pull the last checkpoint (`checkpoint_sync.py --pull`) before running
   `train.py --resume_from ...`.
6. Check `df -h /kaggle/working` — make sure there's room before starting
   a long training run.
7. Set an alarm for ~11h30 from the start — Kaggle doesn't warn before
   hard-killing the session after 12h. Run `checkpoint_sync.py --push`
   before that happens.

---

*Document written during real debugging of Mamba2 2.7B + SAN 2.0 adapter
training on Kaggle (dual T4), September 2026. Library versions at test
time: `torch 2.10.0+cu128`, `transformers` (fresh, from the Kaggle
image), `mamba-ssm 2.3.2.post1`, `causal-conv1d 1.7.0`. The behaviors
described in sections 2-3 may change in future `transformers` versions —
worth re-verifying if something described here stops matching reality.*
