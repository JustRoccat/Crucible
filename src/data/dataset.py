from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ..tokenizer import ProductionTokenizer


class TokenizedTextDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: ProductionTokenizer,
        seq_len: int = 128,
        chunk_chars: int = 20000000,
        force_rebuild_cache: bool = False,
    ):
        file_path = Path(file_path)
        cache_path = self._cache_path_for(file_path, tokenizer)
        if force_rebuild_cache or not cache_path.exists():
            self._build_cache(file_path, tokenizer, cache_path, chunk_chars)
        self.data = np.load(cache_path, mmap_mode="r")
        self.seq_len = seq_len
        if len(self.data) <= seq_len + 1:
            raise ValueError(
                f"After tokenization the corpus only has {len(self.data)} tokens, but seq_len={seq_len} requires at least seq_len+2. Increase the corpus or decrease seq_len."
            )

    @staticmethod
    def _cache_path_for(file_path: Path, tokenizer: ProductionTokenizer) -> Path:
        stat = file_path.stat()
        tag = f"{getattr(tokenizer, 'backend_name', 'unk')}_{tokenizer.vocab_size}_{stat.st_size}_{int(stat.st_mtime)}"
        h = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
        return file_path.with_suffix(f".ids.{h}.npy")

    @staticmethod
    def _build_cache(
        file_path: Path,
        tokenizer: ProductionTokenizer,
        cache_path: Path,
        chunk_chars: int,
    ) -> None:
        print(
            f"[dataset] No token cache ({cache_path.name}) — tokenizing {file_path} in chunks of {chunk_chars:,} chars (one-time)..."
        )
        tmp_path = cache_path.with_suffix(".tmp.npy")
        ids_chunks = []
        n_chars = 0
        with open(file_path, encoding="utf-8") as f:
            while True:
                text = f.read(chunk_chars)
                if not text:
                    break
                n_chars += len(text)
                chunk_ids = tokenizer.encode(text)
                ids_chunks.append(np.asarray(chunk_ids, dtype=np.uint32))
                print(
                    f"[dataset]   processed {n_chars:,} chars ({sum((len(c) for c in ids_chunks)):,} tokens so far)"
                )
        if not ids_chunks:
            raise ValueError(f"File {file_path} is empty — nothing to tokenize.")
        all_ids = np.concatenate(ids_chunks)
        del ids_chunks
        np.save(tmp_path, all_ids)
        tmp_path.replace(cache_path)
        print(
            f"[dataset] Cache saved: {cache_path} ({len(all_ids):,} tokens, {all_ids.nbytes / 1000000.0:.1f} MB on disk)."
        )

    def __len__(self) -> int:
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.seq_len + 1]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        return (chunk[:-1], chunk[1:])


class SyntheticDataset(Dataset):

    def __init__(
        self,
        vocab_size: int = 8000,
        seq_len: int = 128,
        n_samples: int = 1000,
        seed: int = 0,
    ):
        g = torch.Generator().manual_seed(seed)
        base_pattern = torch.randint(0, vocab_size, (seq_len // 4 + 1,), generator=g)
        self.samples = []
        for _ in range(n_samples):
            noise = torch.randint(0, vocab_size, (seq_len + 1,), generator=g)
            pattern_repeated = base_pattern.repeat(seq_len // len(base_pattern) + 2)[
                : seq_len + 1
            ]
            mix_mask = torch.rand(seq_len + 1, generator=g) < 0.7
            seq = torch.where(mix_mask, pattern_repeated, noise)
            self.samples.append(seq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        chunk = self.samples[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return (x, y)
