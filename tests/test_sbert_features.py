"""SBERT feature shapes and evidence pooling, using the fake encoder only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src.features_sbert import (
    build_features,
    embed_evidence,
    encode_texts,
    mean_pool_embeddings,
    pair_lists_from_frame,
    split_evidence_text,
)


def test_build_features_shapes_and_full_values():
    claim = np.array([[1.0, 0.0], [0.5, 0.5]])
    evidence = np.array([[0.0, 2.0], [1.0, -1.0]])
    flag = np.array([1, 0])
    full = build_features(claim, evidence, flag, "full")
    claim_only = build_features(claim, evidence, flag, "claim_only")
    evidence_only = build_features(claim, evidence, flag, "evidence_only")
    assert full.shape == (2, 9)
    assert claim_only.shape == (2, 2)
    assert evidence_only.shape == (2, 3)
    expected = np.array([1.0, 0.0, 0.0, 2.0, 1.0, 2.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(full[0], expected)
    np.testing.assert_allclose(claim_only, claim)
    np.testing.assert_allclose(evidence_only[:, :-1], evidence)
    np.testing.assert_array_equal(evidence_only[:, -1], flag)


def test_mean_pool_groups_fake_encoder_rows():
    frame = pd.DataFrame({
        "evidence_pairs": [
            ["alpha pair", "beta pair"],
            [C.NO_EVIDENCE_TOKEN],
            ["one", "two", "three"],
        ]
    })
    lists = pair_lists_from_frame(frame)
    flat = [pair for pairs in lists for pair in pairs]
    encoded = encode_texts(flat, "unused-model", fake=True)
    pooled = mean_pool_embeddings(encoded, [len(pairs) for pairs in lists])
    assert pooled.shape == (3, 384)
    np.testing.assert_allclose(pooled[0], encoded[:2].mean(axis=0))
    np.testing.assert_allclose(pooled[1], encoded[2])
    np.testing.assert_allclose(pooled[2], encoded[3:6].mean(axis=0))
    again = encode_texts(flat, "unused-model", fake=True)
    np.testing.assert_array_equal(encoded, again)


def test_embed_evidence_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "EMBED_DIR", tmp_path)
    frame = pd.DataFrame({
        "evidence_pairs": [["left", "right"], [C.NO_EVIDENCE_TOKEN]],
    })
    first = embed_evidence(frame, "unit-model", split="toy", fake=True)
    second = embed_evidence(frame, "unit-model", split="toy", fake=True)
    np.testing.assert_array_equal(first, second)
    cached = list(tmp_path.glob("*_fake_toy_evidence.npy"))
    assert len(cached) == 1


def test_split_evidence_text_placeholder():
    text, pairs, flag = split_evidence_text("   ")
    assert text == C.NO_EVIDENCE_TOKEN
    assert pairs == [C.NO_EVIDENCE_TOKEN]
    assert flag == 0
    text, pairs, flag = split_evidence_text("Q: a A: b [SEP] Q: c A: d")
    assert pairs == ["Q: a A: b", "Q: c A: d"]
    assert flag == 1
    assert " [SEP] " in text
