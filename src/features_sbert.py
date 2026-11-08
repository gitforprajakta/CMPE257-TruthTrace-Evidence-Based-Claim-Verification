"""Sentence-BERT embeddings and the feature vectors used by logistic regression.

`build_features` is the single definition of the three input modes. The demo
calls the same helpers instead of reimplementing pooling or concatenation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from . import config as C
from .utils import get_logger

logger = get_logger(__name__)

FAKE_DIM = 384


def model_slug(model_name: str, fake: bool = False) -> str:
    """Cache key: model name with '/' replaced by '_'. Fake runs get a suffix."""
    slug = model_name.replace("/", "_")
    if fake:
        # Keep hash vectors out of the real encoder cache. A fake file with the
        # same row count would otherwise be reused as if it were MiniLM.
        slug = f"{slug}_fake"
    return slug


def resolve_device(device: Optional[str] = None) -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if device:
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _fake_vector(text: str, dim: int = FAKE_DIM) -> np.ndarray:
    """Deterministic unit vector from a SHA-256 seed. Same text, same vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype(np.float64)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def encode_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 64,
    device: Optional[str] = None,
    fake: bool = False,
    encoder=None,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode texts with a SentenceTransformer, or with the fake hash encoder.

    Embeddings are L2-normalised. `encoder`, when given, is used as-is so the
    demo can pass a cached model instead of loading another copy.
    """
    cleaned = ["" if text is None else str(text) for text in texts]
    if fake:
        if not cleaned:
            return np.zeros((0, FAKE_DIM), dtype=np.float64)
        return np.vstack([_fake_vector(text) for text in cleaned])

    if encoder is None:
        from sentence_transformers import SentenceTransformer

        chosen = resolve_device(device)
        logger.info("Loading %s on %s", model_name, chosen)
        encoder = SentenceTransformer(model_name, device=chosen)
    if not cleaned:
        dim = int(encoder.get_sentence_embedding_dimension())
        return np.zeros((0, dim), dtype=np.float64)
    vectors = encoder.encode(
        cleaned,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar and len(cleaned) > 1,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float64)


def _cache_path(model_name: str, split: str, kind: str, fake: bool) -> Path:
    return C.EMBED_DIR / f"{model_slug(model_name, fake=fake)}_{split}_{kind}.npy"


def cache_is_ready(
    model_name: str,
    split: str,
    kind: str,
    n_rows: int,
    fake: bool = False,
    force: bool = False,
) -> bool:
    """True when a cache file exists and has one row per claim."""
    if force:
        return False
    path = _cache_path(model_name, split, kind, fake)
    if not path.exists():
        return False
    cached = np.load(path, mmap_mode="r")
    return int(cached.shape[0]) == n_rows


def _load_cache(path: Path, n_rows: int, force: bool) -> Optional[np.ndarray]:
    if force or not path.exists():
        return None
    cached = np.load(path)
    if cached.shape[0] != n_rows:
        logger.info("Ignoring %s because it has %d rows, expected %d", path.name, cached.shape[0], n_rows)
        return None
    logger.info("Reusing embedding cache %s", path)
    return cached


def _save_cache(path: Path, vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    logger.info("Wrote embedding cache %s shape=%s", path, tuple(vectors.shape))


def embed_claims(
    df: pd.DataFrame,
    model_name: str,
    split: str,
    force: bool = False,
    fake: bool = False,
    batch_size: int = 64,
    device: Optional[str] = None,
    encoder=None,
) -> np.ndarray:
    """Encode `df['claim']`, using the on-disk cache when the row count matches."""
    path = _cache_path(model_name, split, "claim", fake)
    cached = _load_cache(path, len(df), force)
    if cached is not None:
        return cached
    logger.info("Encoding %d claims for split=%s", len(df), split)
    vectors = encode_texts(
        df["claim"].astype(str).tolist(),
        model_name,
        batch_size=batch_size,
        device=device,
        fake=fake,
        encoder=encoder,
    )
    _save_cache(path, vectors)
    return vectors


def pair_list(value) -> list[str]:
    """One claim's evidence strings. Missing evidence becomes the placeholder."""
    if value is None:
        return [C.NO_EVIDENCE_TOKEN]
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return [C.NO_EVIDENCE_TOKEN]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else [C.NO_EVIDENCE_TOKEN]
    try:
        items = list(value)
    except TypeError:
        items = []
    pairs: list[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            pairs.append(text)
    return pairs or [C.NO_EVIDENCE_TOKEN]


def pair_lists_from_frame(df: pd.DataFrame) -> list[list[str]]:
    """Evidence pair lists aligned with the dataframe rows."""
    if "evidence_pairs" not in df.columns:
        raise ValueError("Dataframe is missing evidence_pairs")
    return [pair_list(value) for value in df["evidence_pairs"].tolist()]


def mean_pool_embeddings(vectors: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
    """Mean-pool a flat matrix of pair embeddings back to one vector per claim.

    `lengths[i]` is the number of pairs belonging to claim i. Every length is
    at least 1 because missing evidence is stored as a single placeholder.
    """
    flat = np.asarray(vectors, dtype=np.float64)
    group_lengths = np.asarray(list(lengths), dtype=int)
    if group_lengths.size == 0:
        dim = flat.shape[1] if flat.ndim == 2 else FAKE_DIM
        return np.zeros((0, dim), dtype=np.float64)
    if np.any(group_lengths < 1):
        raise ValueError("every claim needs at least one evidence string to pool")
    if flat.shape[0] != int(group_lengths.sum()):
        raise ValueError(
            f"Got {flat.shape[0]} embeddings for {int(group_lengths.sum())} evidence strings"
        )
    starts = np.zeros(group_lengths.shape[0], dtype=int)
    if group_lengths.shape[0] > 1:
        starts[1:] = np.cumsum(group_lengths)[:-1]
    pooled = np.add.reduceat(flat, starts, axis=0)
    return pooled / group_lengths[:, None]


def embed_evidence(
    df: pd.DataFrame,
    model_name: str,
    split: str,
    force: bool = False,
    fake: bool = False,
    batch_size: int = 64,
    device: Optional[str] = None,
    encoder=None,
) -> np.ndarray:
    """Encode each evidence pair, then mean-pool to one vector per claim."""
    path = _cache_path(model_name, split, "evidence", fake)
    cached = _load_cache(path, len(df), force)
    if cached is not None:
        return cached
    lists = pair_lists_from_frame(df)
    lengths = [len(pairs) for pairs in lists]
    flat_texts = [pair for pairs in lists for pair in pairs]
    logger.info(
        "Encoding %d evidence strings (%d claims) for split=%s",
        len(flat_texts), len(df), split,
    )
    flat = encode_texts(
        flat_texts,
        model_name,
        batch_size=batch_size,
        device=device,
        fake=fake,
        encoder=encoder,
    )
    pooled = mean_pool_embeddings(flat, lengths)
    _save_cache(path, pooled)
    return pooled


def split_evidence_text(evidence: str) -> tuple[str, list[str], int]:
    """Split user evidence on ' [SEP] '. Empty input uses the placeholder."""
    text = "" if evidence is None else str(evidence).strip()
    if not text or text == C.NO_EVIDENCE_TOKEN:
        return C.NO_EVIDENCE_TOKEN, [C.NO_EVIDENCE_TOKEN], 0
    pairs = [part.strip() for part in text.split(" [SEP] ") if part.strip()]
    if not pairs:
        return C.NO_EVIDENCE_TOKEN, [C.NO_EVIDENCE_TOKEN], 0
    return " [SEP] ".join(pairs), pairs, 1


def build_features(
    claim_emb: np.ndarray,
    evid_emb: np.ndarray,
    has_evidence,
    mode: str,
) -> np.ndarray:
    """Build the logistic-regression matrix for one input mode.

    `full` is `[u, v, |u-v|, u*v, has_evidence]`. `claim_only` is `[u]`.
    `evidence_only` is `[v, has_evidence]`.
    """
    if mode not in C.INPUT_MODES:
        raise ValueError(f"Unknown input mode {mode!r}. Expected one of {C.INPUT_MODES}.")
    claim = np.asarray(claim_emb, dtype=np.float64)
    evidence = np.asarray(evid_emb, dtype=np.float64)
    if claim.ndim != 2 or evidence.ndim != 2:
        raise ValueError("claim and evidence embeddings must be 2-D arrays")
    if claim.shape[0] != evidence.shape[0]:
        raise ValueError("claim and evidence embeddings have different row counts")
    flag = np.asarray(has_evidence, dtype=np.float64).reshape(-1, 1)
    if flag.shape[0] != claim.shape[0]:
        raise ValueError("has_evidence length does not match the embeddings")
    if mode == "claim_only":
        return claim.copy()
    if mode == "evidence_only":
        return np.hstack([evidence, flag])
    if claim.shape[1] != evidence.shape[1]:
        raise ValueError("full mode requires claim and evidence embeddings with the same width")
    return np.hstack([claim, evidence, np.abs(claim - evidence), claim * evidence, flag])
