"""TF-IDF pipeline tests on a tiny repeated corpus (min_df=2)."""
from __future__ import annotations

import joblib
import pandas as pd

from src import config as C
from src.features_tfidf import build_pipeline, input_frame


def _toy_frame() -> pd.DataFrame:
    rows = []
    for label in C.LABELS:
        token = label.split()[0].lower()
        for _ in range(4):
            rows.append({
                "claim": f"{token} claim alpha alpha repeated",
                "evidence_text": f"{token} evidence beta beta repeated",
                "has_evidence": 1,
                "label": label,
            })
    return pd.DataFrame(rows)


def test_pipeline_fits_all_modes_and_unpickles(tmp_path):
    frame = _toy_frame()
    features = input_frame(frame)
    one_row = features.iloc[[0]].copy()
    for mode in C.INPUT_MODES:
        pipeline = build_pipeline(mode, ngram_range=(1, 1))
        pipeline.fit(features, frame["label"])
        proba = pipeline.predict_proba(one_row)
        assert proba.shape == (1, len(pipeline.named_steps["clf"].classes_))
        path = tmp_path / f"{mode}.joblib"
        joblib.dump(pipeline, path)
        loaded = joblib.load(path)
        loaded_proba = loaded.predict_proba(one_row)
        assert loaded_proba.shape == proba.shape
        assert loaded.predict(one_row).shape == (1,)
