"""Decision-support demo for the two full-mode claim verifiers.

This is not a source of truth. It scores one claim with the saved TF-IDF
pipeline and the saved SBERT pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from src import config as C
from src.evaluate import align_proba
from src.features_sbert import (
    build_features,
    encode_texts,
    mean_pool_embeddings,
    resolve_device,
    split_evidence_text,
)
from src.features_tfidf import input_frame
from src.utils import read_json

DISCLAIMER = (
    "This is a decision-support prototype trained on AVeriTeC. "
    "It is not a source of truth."
)


def load_tfidf_model():
    """Return the full-mode TF-IDF pipeline, or None when the file is absent."""
    path = C.MODELS_DIR / "tfidf_lr_full.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def load_sbert_artifacts():
    """Return the SBERT pipeline and its sidecar, or (None, None)."""
    model_path = C.MODELS_DIR / "sbert_lr_full.joblib"
    meta_path = C.MODELS_DIR / "sbert_lr_full.json"
    if not model_path.exists() or not meta_path.exists():
        return None, None
    return joblib.load(model_path), read_json(meta_path)


def load_encoder(model_name: str):
    """Load one SentenceTransformer. The app caches this with st.cache_resource."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=resolve_device(None))


def example_catalog() -> pd.DataFrame:
    """Claims the dropdown can load. Empty when the processed test split is absent."""
    if not C.TEST_PARQUET.exists():
        return pd.DataFrame(columns=["claim_id", "claim", "evidence_text", "label"])
    test = pd.read_parquet(C.TEST_PARQUET)
    return test.loc[:, ["claim_id", "claim", "evidence_text", "label"]].copy()


def predict_both(claim: str, evidence: str, tfidf_model, sbert_model, sbert_meta, encoder) -> dict:
    """Score one claim with whichever saved models are loaded."""
    evidence_text, pairs, has_evidence = split_evidence_text(evidence)
    result: dict = {"tfidf": None, "sbert": None}
    if tfidf_model is not None:
        frame = input_frame(pd.DataFrame([{
            "claim": claim,
            "evidence_text": evidence_text,
            "has_evidence": has_evidence,
        }]))
        label = str(tfidf_model.predict(frame)[0])
        aligned = align_proba(tfidf_model.predict_proba(frame), list(tfidf_model.classes_))
        result["tfidf"] = {"label": label, "proba": aligned[0]}
    if sbert_model is not None and sbert_meta is not None and encoder is not None:
        model_name = sbert_meta["sbert_model"]
        claim_emb = encode_texts(
            [claim], model_name, encoder=encoder, show_progress_bar=False,
        )
        pair_emb = encode_texts(
            pairs, model_name, encoder=encoder, show_progress_bar=False,
        )
        evidence_emb = mean_pool_embeddings(pair_emb, [len(pairs)])
        features = build_features(
            claim_emb, evidence_emb, [has_evidence], sbert_meta["feature_mode"],
        )
        label = str(sbert_model.predict(features)[0])
        aligned = align_proba(sbert_model.predict_proba(features), list(sbert_model.classes_))
        result["sbert"] = {"label": label, "proba": aligned[0]}
    return result


def _show_model(title: str, prediction: dict | None, missing_message: str) -> None:
    st.subheader(title)
    if prediction is None:
        st.info(missing_message)
        return
    st.write(f"Predicted verdict: **{prediction['label']}**")
    chart = pd.DataFrame({
        "probability": prediction["proba"],
    }, index=[C.SHORT_LABELS[label] for label in C.LABELS])
    st.bar_chart(chart)


def main() -> None:
    st.set_page_config(page_title="TruthTrace demo", layout="wide")
    st.title("TruthTrace")
    st.markdown(
        "<div style='padding:0.8rem 1rem;border:2px solid #9a3412;background:#fff7ed;"
        "border-radius:8px;font-weight:600;margin-bottom:1rem;'>"
        f"{DISCLAIMER}</div>",
        unsafe_allow_html=True,
    )
    st.caption("Paste a claim and the evidence, or load an example from the AVeriTeC dev split.")

    tfidf_model = cached_tfidf()
    sbert_model, sbert_meta = cached_sbert()
    catalog = example_catalog()
    options = ["Custom"]
    lookup = {}
    for _, row in catalog.iterrows():
        label = f"{int(row['claim_id'])}: {str(row['claim'])[:90]}"
        options.append(label)
        lookup[label] = row

    choice = st.selectbox("Load an example", options)
    if choice == "Custom":
        claim_value = ""
        evidence_value = ""
    else:
        selected = lookup[choice]
        claim_value = str(selected["claim"])
        evidence_value = str(selected["evidence_text"])
        st.caption(f"Recorded label in the dataset: {selected['label']}")

    claim = st.text_area("Claim", value=claim_value, height=120, key=f"claim-{choice}")
    evidence = st.text_area(
        "Evidence (QA pairs separated by [SEP], or free text)",
        value=evidence_value,
        height=220,
        key=f"evidence-{choice}",
    )
    if st.button("Check claim", type="primary"):
        if not claim.strip():
            st.warning("Enter a claim before checking it.")
            return
        encoder = None
        if sbert_model is not None and sbert_meta is not None:
            encoder = cached_encoder(sbert_meta["sbert_model"])
        scored = predict_both(claim, evidence, tfidf_model, sbert_model, sbert_meta, encoder)
        left, right = st.columns(2)
        with left:
            _show_model("TF-IDF + logistic regression", scored["tfidf"], "TF-IDF model not available")
        with right:
            _show_model("SBERT + logistic regression", scored["sbert"], "SBERT model not available")


@st.cache_resource
def cached_tfidf():
    return load_tfidf_model()


@st.cache_resource
def cached_sbert():
    return load_sbert_artifacts()


@st.cache_resource
def cached_encoder(model_name: str):
    return load_encoder(model_name)


main()
