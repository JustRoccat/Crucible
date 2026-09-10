import time
import numpy as np
import pytest
import torch
from src.data.dataset import SyntheticDataset, TokenizedTextDataset
from src.tokenizer import ProductionTokenizer


def _byte_tokenizer():
    return ProductionTokenizer.from_config({"mode": "byte_fallback"})


def test_next_token_shift_is_correct():
    tok = _byte_tokenizer()
    text = "".join((chr(ord("a") + i % 26) for i in range(500)))

    class _FakeDataset(TokenizedTextDataset):

        def __init__(self):
            self.data = np.array(tok.encode(text), dtype=np.uint32)
            self.seq_len = 16

    ds = _FakeDataset()
    x, y = ds[10]
    assert torch.equal(x[1:], y[:-1])
    assert y[-1].item() == ds.data[10 + ds.seq_len]


def test_build_and_reload_cache(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "the quick brown fox jumps over the lazy dog " * 50, encoding="utf-8"
    )
    tok = _byte_tokenizer()
    ds1 = TokenizedTextDataset(str(corpus), tokenizer=tok, seq_len=8, chunk_chars=64)
    cache_files = list(tmp_path.glob("*.ids.*.npy"))
    assert len(cache_files) == 1
    mtime_before = cache_files[0].stat().st_mtime
    ds2 = TokenizedTextDataset(str(corpus), tokenizer=tok, seq_len=8, chunk_chars=64)
    assert cache_files[0].stat().st_mtime == mtime_before
    assert len(ds1) == len(ds2)
    assert np.array_equal(ds1.data[:], ds2.data[:])


def test_editing_source_file_invalidates_cache(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("original content " * 20, encoding="utf-8")
    tok = _byte_tokenizer()
    ds1 = TokenizedTextDataset(str(corpus), tokenizer=tok, seq_len=8, chunk_chars=64)
    old_cache_files = set(tmp_path.glob("*.ids.*.npy"))
    time.sleep(1.05)
    corpus.write_text(
        "completely different content, much longer than before " * 20, encoding="utf-8"
    )
    ds2 = TokenizedTextDataset(str(corpus), tokenizer=tok, seq_len=8, chunk_chars=64)
    new_cache_files = set(tmp_path.glob("*.ids.*.npy"))
    assert (
        new_cache_files != old_cache_files
    ), "cache was not rebuilt after editing the source file"
    assert len(ds1) != len(ds2) or not np.array_equal(ds1.data[:], ds2.data[:])


def test_force_rebuild_cache_flag(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("some text here " * 30, encoding="utf-8")
    tok = _byte_tokenizer()
    TokenizedTextDataset(str(corpus), tokenizer=tok, seq_len=8, chunk_chars=64)
    cache_path = next(tmp_path.glob("*.ids.*.npy"))
    cache_path.unlink()
    cache_path.write_bytes(b"not a real npy file")
    ds = TokenizedTextDataset(
        str(corpus), tokenizer=tok, seq_len=8, chunk_chars=64, force_rebuild_cache=True
    )
    assert len(ds) > 0


def test_raises_on_corpus_too_short_for_seq_len(tmp_path):
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("hi", encoding="utf-8")
    tok = _byte_tokenizer()
    with pytest.raises(ValueError):
        TokenizedTextDataset(str(corpus), tokenizer=tok, seq_len=128, chunk_chars=64)


def test_raises_on_empty_file(tmp_path):
    corpus = tmp_path / "empty.txt"
    corpus.write_text("", encoding="utf-8")
    tok = _byte_tokenizer()
    with pytest.raises(ValueError):
        TokenizedTextDataset(str(corpus), tokenizer=tok, seq_len=8, chunk_chars=64)


def test_chunked_tokenization_matches_single_shot_for_byte_backend(tmp_path):
    text = "abcdefghij" * 37
    tok = _byte_tokenizer()
    single_shot = np.array(tok.encode(text), dtype=np.uint32)
    corpus = tmp_path / "c.txt"
    corpus.write_text(text, encoding="utf-8")
    ds = TokenizedTextDataset(str(corpus), tokenizer=tok, seq_len=8, chunk_chars=17)
    assert np.array_equal(ds.data[:], single_shot)


def test_synthetic_dataset_shapes_and_shift():
    ds = SyntheticDataset(vocab_size=32, seq_len=20, n_samples=5, seed=0)
    assert len(ds) == 5
    x, y = ds[0]
    assert x.shape == (20,)
    assert y.shape == (20,)


def test_synthetic_dataset_deterministic_with_seed():
    ds1 = SyntheticDataset(vocab_size=32, seq_len=10, n_samples=3, seed=42)
    ds2 = SyntheticDataset(vocab_size=32, seq_len=10, n_samples=3, seed=42)
    for i in range(3):
        x1, y1 = ds1[i]
        x2, y2 = ds2[i]
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
