from __future__ import annotations
import os


class _ByteFallbackBackend:
    name = "byte_fallback"

    def __init__(self):
        self._vocab_size = 256
        self.pad_id = None
        self.eos_id = None
        self.bos_id = None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int]) -> str:
        return bytes((int(i) & 255 for i in ids)).decode("utf-8", errors="replace")


class _TiktokenBackend:
    name = "tiktoken"

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)
        self._vocab_size = self._enc.n_vocab
        self.eos_id = self._enc.eot_token
        self.bos_id = None
        self.pad_id = self._enc.eot_token

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(list((int(i) for i in ids)))


class _HFTokenizersBackend:
    name = "hf_tokenizers"

    def __init__(
        self, pretrained_repo: str | None = None, local_path: str | None = None
    ):
        from tokenizers import Tokenizer

        if local_path is not None:
            self._tok = Tokenizer.from_file(local_path)
            self._tok.no_truncation()
        elif pretrained_repo is not None:
            self._tok = Tokenizer.from_pretrained(pretrained_repo)
            self._tok.no_truncation()
        else:
            raise ValueError("Provide either `pretrained_repo` or `local_path`.")
        self._vocab_size = self._tok.get_vocab_size()
        self.pad_id = self._safe_id(["<pad>", "<|pad|>", "[PAD]"])
        self.eos_id = self._safe_id(["<|endoftext|>", "</s>", "<|end|>", "[EOS]"])
        self.bos_id = self._safe_id(["<|startoftext|>", "<s>", "[BOS]"])

    def _safe_id(self, candidates: list[str]) -> int | None:
        for c in candidates:
            tid = self._tok.token_to_id(c)
            if tid is not None:
                return tid
        return None

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode([int(i) for i in ids])


def train_bpe_tokenizer(
    corpus_path: str,
    vocab_size: int = 8000,
    save_path: str = "tokenizer.json",
    special_tokens: list[str] | None = None,
) -> str:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    special_tokens = special_tokens or ["<pad>", "<|endoftext|>", "<unk>"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tok.train([corpus_path], trainer)
    tok.save(save_path)
    return save_path


class ProductionTokenizer:

    def __init__(self, backend):
        self._backend = backend

    @property
    def vocab_size(self) -> int:
        return self._backend.vocab_size

    @property
    def pad_id(self):
        return self._backend.pad_id

    @property
    def eos_id(self):
        return self._backend.eos_id

    @property
    def bos_id(self):
        return self._backend.bos_id

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def encode(self, text: str) -> list[int]:
        return self._backend.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._backend.decode(ids)

    @classmethod
    def from_config(cls, cfg: dict) -> ProductionTokenizer:
        mode = cfg.get("mode", "byte_fallback")
        if mode == "tiktoken":
            try:
                backend = _TiktokenBackend(cfg.get("tiktoken_encoding", "cl100k_base"))
            except ImportError:
                print("[tokenizer] `tiktoken` unavailable -> falling back to bytes.")
                backend = _ByteFallbackBackend()
        elif mode == "hf_pretrained":
            try:
                backend = _HFTokenizersBackend(pretrained_repo=cfg["hf_repo"])
            except ImportError:
                print(
                    "[tokenizer] `tokenizers` library unavailable -> falling back to bytes."
                )
                backend = _ByteFallbackBackend()
        elif mode == "train_bpe":
            cache_path = cfg.get("cache_path", "checkpoints/tokenizer.json")
            try:
                if not os.path.exists(cache_path):
                    corpus = cfg["train_corpus"]
                    vocab_size = int(cfg.get("train_vocab_size", 8000))
                    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                    print(
                        f"[tokenizer] Training BPE (vocab={vocab_size}) on {corpus} ..."
                    )
                    train_bpe_tokenizer(
                        corpus, vocab_size=vocab_size, save_path=cache_path
                    )
                backend = _HFTokenizersBackend(local_path=cache_path)
            except ImportError:
                print(
                    "[tokenizer] `tokenizers` library unavailable -> falling back to bytes."
                )
                backend = _ByteFallbackBackend()
        elif mode == "byte_fallback":
            backend = _ByteFallbackBackend()
        else:
            raise ValueError(f"Unknown tokenizer mode: {mode}")
        return cls(backend)
