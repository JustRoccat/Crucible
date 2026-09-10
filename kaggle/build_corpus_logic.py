#!/usr/bin/env python3
import argparse
import ast
import gzip
import hashlib
import heapq
import itertools
import os
import random
import re
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
from datasets import load_dataset

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
EOS_MARKER = "<|endoftext|>"
STACK_REPO = "bigcode/the-stack-dedup"
LANG_CONFIG = {
    "python": dict(
        data_dir="data/python", budget_mb=2000, min_score=6, max_scan=400000
    ),
    "c++": dict(data_dir="data/cpp", budget_mb=2000, min_score=6, max_scan=400000),
    "rust": dict(data_dir="data/rust", budget_mb=2000, min_score=6, max_scan=400000),
}
MIN_LEN = 200
MAX_LEN = 30000
HEAP_CAP = 250000


def score_python(code: str) -> float:
    try:
        tree = ast.parse(code)
    except Exception:
        return -1.0
    score = 0.0
    has_test_import = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            score += 3.0
        elif isinstance(node, ast.Compare):
            score += 0.4
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            score += 0.5
            if node.returns is not None:
                score += 1.0
            if any((a.annotation is not None for a in node.args.args)):
                score += 1.0
            if ast.get_docstring(node):
                score += 0.5
        elif isinstance(node, ast.ClassDef):
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            if "TestCase" in bases:
                score += 8.0
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("unittest", "pytest"):
                    has_test_import = True
        elif isinstance(node, ast.ImportFrom):
            if node.module in ("unittest", "pytest"):
                has_test_import = True
        elif isinstance(node, ast.Raise):
            score += 0.5
    if has_test_import:
        score += 6.0
    if re.search("@pytest\\.(fixture|mark)", code):
        score += 4.0
    return score


CPP_PATTERNS = [
    ("\\bstatic_assert\\s*\\(", 4.0),
    ("\\bassert\\s*\\(", 2.5),
    ("\\bTEST(_F|_P)?\\s*\\(", 6.0),
    ("\\bTEST_CASE\\s*\\(", 6.0),
    ("\\bREQUIRE\\s*\\(", 3.0),
    ("\\bCHECK\\s*\\(", 2.0),
    ("\\bEXPECT_\\w+\\s*\\(", 2.5),
    ("//\\s*(pre|post|invariant)\\s*:", 3.0),
    ("\\bnoexcept\\b", 0.5),
    ("\\btemplate\\s*<", 1.0),
    ("\\bstd::(optional|variant|expected)\\b", 1.5),
]
RUST_PATTERNS = [
    ("#\\[test\\]", 6.0),
    ("#\\[cfg\\(test\\)\\]", 4.0),
    ("\\bassert_eq!\\s*\\(", 3.0),
    ("\\bassert_ne!\\s*\\(", 3.0),
    ("\\bassert!\\s*\\(", 2.5),
    ("\\bdebug_assert\\w*!\\s*\\(", 2.0),
    ("\\bResult<", 1.0),
    ("\\bOption<", 0.5),
    ("///", 0.3),
    ("\\bunwrap_or\\b|\\bexpect\\(", 0.5),
]


def score_regex(code: str, patterns) -> float:
    score = 0.0
    for pattern, weight in patterns:
        n = len(re.findall(pattern, code))
        score += n * weight
    return score


def score_for_lang(lang: str, code: str) -> float:
    if lang == "python":
        return score_python(code)
    elif lang == "c++":
        return score_regex(code, CPP_PATTERNS)
    elif lang == "rust":
        return score_regex(code, RUST_PATTERNS)
    raise ValueError(lang)


def fmt_block(lang: str, path: str, code: str) -> str:
    return f"### Language\n{lang}\n\n### Path\n{path}\n\n### Code\n{code}".rstrip()


def collect_language(lang: str, cfg: dict) -> list:
    print(
        f"\n=== {lang} | data_dir={cfg['data_dir']} | budget={cfg['budget_mb']} MB | min_score={cfg['min_score']} | max_scan={cfg['max_scan']} ==="
    )
    ds = load_dataset(
        STACK_REPO, data_dir=cfg["data_dir"], split="train", streaming=True
    )
    ds = ds.shuffle(seed=RANDOM_SEED, buffer_size=20000)
    heap = []
    counter = itertools.count()
    seen_hashes = set()
    scanned = 0
    accepted = 0
    for row in ds:
        scanned += 1
        if scanned > cfg["max_scan"]:
            print(f"  [STOP] reached max_scan={cfg['max_scan']} samples")
            break
        code = row.get("content") or ""
        if not MIN_LEN <= len(code) <= MAX_LEN:
            continue
        h = hashlib.md5(code.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        score = score_for_lang(lang, code)
        if score < cfg["min_score"]:
            continue
        accepted += 1
        path = row.get("max_stars_repo_path") or row.get("path") or "unknown"
        item = (score, next(counter), path, code)
        if len(heap) < HEAP_CAP:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)
        if scanned % 20000 == 0:
            print(f"  ...scanned={scanned} accepted={accepted} heap_size={len(heap)}")
    print(
        f"  Scanned {scanned}, accepted (score>=min) {accepted}, in the final top-K heap {len(heap)}"
    )
    heap.sort(key=lambda x: -x[0])
    return heap


def write_language_budget(lang: str, heap: list, budget_mb: int, out_f) -> int:
    budget_bytes = budget_mb * 1024 * 1024
    written = 0
    n_blocks = 0
    for score, _, path, code in heap:
        block = fmt_block(lang, path, code)
        nbytes = len(block.encode("utf-8"))
        if written + nbytes > budget_bytes:
            continue
        out_f.write(block)
        out_f.write(f"\n\n{EOS_MARKER}\n\n")
        written += nbytes
        n_blocks += 1
        if written >= budget_bytes:
            break
    print(
        f"  [{lang}] wrote {n_blocks} blocks, {written / 1000000.0:.1f} MB (budget {budget_mb} MB){(' [BELOW BUDGET]' if written < budget_bytes * 0.8 else '')}"
    )
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="kaggle/corpus_logic.txt")
    ap.add_argument("--langs", nargs="+", default=list(LANG_CONFIG.keys()))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    all_blocks = []
    total_written = 0
    for lang in args.langs:
        cfg = LANG_CONFIG[lang]
        heap = collect_language(lang, cfg)
        for score, _, path, code in heap:
            all_blocks.append((lang, path, code, score))
    random.shuffle(all_blocks)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for lang in args.langs:
            cfg = LANG_CONFIG[lang]
            lang_items = [b for b in all_blocks if b[0] == lang]
            heap_like = [
                (score, i, path, code)
                for i, (l, path, code, score) in enumerate(lang_items)
            ]
            heap_like.sort(key=lambda x: -x[0])
            total_written += write_language_budget(
                lang, heap_like, cfg["budget_mb"], out_f
            )
    final_mb = os.path.getsize(args.output) / 1000000.0
    print(
        f"\nDONE: {args.output} -> {final_mb:.1f} MB (target: {sum((c['budget_mb'] for c in LANG_CONFIG.values()))} MB)"
    )
    print("Set in configs/default.yaml: train.train_file: " + args.output)


if __name__ == "__main__":
    main()
