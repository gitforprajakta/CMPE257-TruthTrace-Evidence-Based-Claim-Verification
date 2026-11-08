"""Sample misclassified claims and tag a likely reason. Manual reasons stay blank."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C
from .evaluate import load_predictions
from .utils import get_logger, set_seed

logger = get_logger(__name__)

ERROR_COLUMNS = [
    "claim_id",
    "claim",
    "true",
    "pred_tfidf",
    "pred_sbert",
    "evidence_text",
    "has_evidence",
    "n_qa",
    "claim_len",
    "auto_reason",
    "manual_reason",
]
SHORTCUT_PHRASES = ("no answer could be found", "no answer")
CONFLICTING = "Conflicting Evidence/Cherrypicking"
N_ERRORS = 15
N_DISAGREE = 10


def auto_reason(row: pd.Series, predicted: list[str]) -> str:
    """First matching rule wins."""
    if int(row["has_evidence"]) == 0:
        return "no_evidence"
    evidence = str(row["evidence_text"]).casefold()
    if str(row["true"]) == "Not Enough Evidence" and any(phrase in evidence for phrase in SHORTCUT_PHRASES):
        return "shortcut_phrase"
    if str(row["true"]) == CONFLICTING or any(label == CONFLICTING for label in predicted):
        return "conflicting_confusion"
    if int(row["n_qa"]) <= 1:
        return "thin_evidence"
    if int(row["claim_len"]) < 8:
        return "short_claim"
    return "other"


def _predicted_labels(row: pd.Series) -> list[str]:
    labels = []
    for column in ("pred_tfidf", "pred_sbert"):
        value = row.get(column, "")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        text = str(value).strip()
        if text:
            labels.append(text)
    return labels


def stratified_sample(frame: pd.DataFrame, n: int, seed: int = C.SEED) -> pd.DataFrame:
    """Sample up to `n` rows, spread across the true classes with seed `seed`."""
    if frame.empty or n <= 0:
        return frame.iloc[0:0]
    present = []
    for label in C.LABELS:
        group = frame[frame["true"] == label]
        if not group.empty:
            present.append(group)
    if not present:
        return frame.iloc[0:0]
    total = sum(len(group) for group in present)
    target = min(n, total)
    raw = [target * len(group) / total for group in present]
    quotas = [min(len(group), int(np.floor(share))) for group, share in zip(present, raw)]
    leftover = target - sum(quotas)
    order = sorted(
        range(len(present)),
        key=lambda index: (raw[index] - quotas[index], len(present[index])),
        reverse=True,
    )
    for index in order:
        if leftover <= 0:
            break
        room = len(present[index]) - quotas[index]
        take = min(room, leftover)
        quotas[index] += take
        leftover -= take
    rng = np.random.default_rng(seed)
    parts = []
    for group, quota in zip(present, quotas):
        if quota <= 0:
            continue
        chosen = rng.choice(len(group), size=quota, replace=False)
        parts.append(group.iloc[np.sort(chosen)])
    if not parts:
        return frame.iloc[0:0]
    return pd.concat(parts, ignore_index=True).sort_values("claim_id")


def _attach_predictions(test: pd.DataFrame, predictions: pd.DataFrame | None, prefix: str) -> pd.DataFrame:
    frame = test.copy()
    frame["claim_id"] = frame["claim_id"].astype(int)
    if predictions is None:
        frame[f"pred_{prefix}"] = ""
        frame[f"correct_{prefix}"] = np.nan
        return frame
    slim = predictions[["claim_id", "y_pred", "correct"]].copy()
    slim["claim_id"] = slim["claim_id"].astype(int)
    slim = slim.rename(columns={"y_pred": f"pred_{prefix}", "correct": f"correct_{prefix}"})
    return frame.merge(slim, on="claim_id", how="left")


def build_joined(test: pd.DataFrame, tfidf: pd.DataFrame | None, sbert: pd.DataFrame | None) -> pd.DataFrame:
    """Test rows plus both models' predictions. Missing models leave blank predictions."""
    columns = ["claim_id", "claim", "label", "evidence_text", "has_evidence", "n_qa", "claim_len"]
    base = test.loc[:, columns].rename(columns={"label": "true"})
    base = _attach_predictions(base, tfidf, "tfidf")
    base = _attach_predictions(base, sbert, "sbert")
    return base


def _output_frame(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        predicted = _predicted_labels(row)
        evidence = str(row["evidence_text"])
        if len(evidence) > 600:
            evidence = evidence[:600]
        records.append({
            "claim_id": int(row["claim_id"]),
            "claim": str(row["claim"]),
            "true": str(row["true"]),
            "pred_tfidf": "" if pd.isna(row["pred_tfidf"]) else str(row["pred_tfidf"]),
            "pred_sbert": "" if pd.isna(row["pred_sbert"]) else str(row["pred_sbert"]),
            "evidence_text": evidence,
            "has_evidence": int(row["has_evidence"]),
            "n_qa": int(row["n_qa"]),
            "claim_len": int(row["claim_len"]),
            "auto_reason": auto_reason(row, predicted),
            "manual_reason": "",
        })
    return pd.DataFrame(records, columns=ERROR_COLUMNS)


def _sample_disagreements(joined: pd.DataFrame, sbert_correct: bool, n: int) -> pd.DataFrame:
    if joined["correct_tfidf"].isna().all() or joined["correct_sbert"].isna().all():
        return joined.iloc[0:0]
    tfidf_ok = joined["correct_tfidf"].astype(float) == 1
    sbert_ok = joined["correct_sbert"].astype(float) == 1
    if sbert_correct:
        mask = sbert_ok & ~tfidf_ok
    else:
        mask = tfidf_ok & ~sbert_ok
    group = joined.loc[mask]
    if group.empty:
        return group
    take = min(n, len(group))
    return group.sample(n=take, random_state=C.SEED).sort_values("claim_id")


def write_summary(tfidf_errors: pd.DataFrame, sbert_errors: pd.DataFrame, disagreements: pd.DataFrame) -> str:
    """Counts per reason and a short list of examples."""
    lines = [
        "# Error analysis",
        "",
        "Reasons are assigned automatically by the first matching rule. "
        "`manual_reason` is left empty for the team to fill in.",
        "",
        "## Counts",
        "",
        "| model | reason | count |",
        "| --- | --- | --- |",
    ]
    for model, frame in (("tfidf_lr", tfidf_errors), ("sbert_lr", sbert_errors)):
        if frame.empty:
            lines.append(f"| {model} | (none sampled) | 0 |")
            continue
        counts = frame["auto_reason"].value_counts()
        for reason, count in counts.items():
            lines.append(f"| {model} | {reason} | {int(count)} |")
    lines.extend(["", "## Examples", ""])
    for model, frame in (("tfidf_lr", tfidf_errors), ("sbert_lr", sbert_errors)):
        lines.append(f"### {model}")
        lines.append("")
        if frame.empty:
            lines.append("No sampled mistakes. The predictions for this model may be missing.")
            lines.append("")
            continue
        for _, row in frame.head(5).iterrows():
            claim = str(row["claim"])
            if len(claim) > 180:
                claim = claim[:180] + "..."
            lines.append(
                f"- claim {int(row['claim_id'])}: true {row['true']}; "
                f"TF-IDF {row['pred_tfidf'] or '—'}; SBERT {row['pred_sbert'] or '—'}; "
                f"reason {row['auto_reason']}. {claim}"
            )
        lines.append("")
    lines.append("## Disagreements")
    lines.append("")
    if disagreements.empty:
        lines.append("No disagreement sample. Both full-mode prediction files are required.")
    else:
        for _, row in disagreements.iterrows():
            claim = str(row["claim"])
            if len(claim) > 160:
                claim = claim[:160] + "..."
            lines.append(
                f"- claim {int(row['claim_id'])} ({row['who_wins']} wins): "
                f"true {row['true']}; TF-IDF {row['pred_tfidf']}; SBERT {row['pred_sbert']}. {claim}"
            )
    lines.append("")
    return "\n".join(lines)


def plot_reason_counts(tfidf_errors: pd.DataFrame, sbert_errors: pd.DataFrame) -> None:
    """Bar chart of automatic reasons for the sampled mistakes."""
    rows = []
    for model, frame in (("TF-IDF", tfidf_errors), ("SBERT", sbert_errors)):
        if frame.empty:
            continue
        for reason, count in frame["auto_reason"].value_counts().items():
            rows.append({"model": model, "reason": str(reason), "count": int(count)})
    path = C.FIGURES_DIR / "error_reason_counts.png"
    if not rows:
        logger.warning("Skipping %s because no error samples were available", path.name)
        return
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=pd.DataFrame(rows), x="reason", y="count", hue="model", ax=axis)
    axis.set_xlabel("Automatic reason")
    axis.set_ylabel("Sampled claims")
    axis.set_title("Error reasons in the sampled mistakes")
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def run() -> None:
    """Write error samples, disagreements, the summary, and the reason chart."""
    set_seed()
    C.ensure_dirs()
    if not C.TEST_PARQUET.exists():
        raise FileNotFoundError("data/processed/test.parquet is missing. Run python -m src.preprocess first.")
    test = pd.read_parquet(C.TEST_PARQUET)
    tfidf = load_predictions("tfidf_lr", "full")
    sbert = load_predictions("sbert_lr", "full")
    if tfidf is None:
        logger.warning("TF-IDF full predictions are missing. Those columns will be empty.")
    if sbert is None:
        logger.warning("SBERT full predictions are missing. Those columns will be empty.")
    joined = build_joined(test, tfidf, sbert)

    samples = {}
    for model, column in (("tfidf_lr", "correct_tfidf"), ("sbert_lr", "correct_sbert")):
        if joined[column].isna().all():
            samples[model] = _output_frame(joined.iloc[0:0])
            continue
        mistakes = joined[joined[column].astype(float) == 0]
        picked = stratified_sample(mistakes, N_ERRORS, seed=C.SEED)
        logger.info("Sampled %d %s mistakes", len(picked), model)
        samples[model] = _output_frame(picked)

    sbert_wins = _sample_disagreements(joined, sbert_correct=True, n=N_DISAGREE)
    tfidf_wins = _sample_disagreements(joined, sbert_correct=False, n=N_DISAGREE)
    disagree_parts = []
    if not sbert_wins.empty:
        tagged = _output_frame(sbert_wins)
        tagged["who_wins"] = "sbert"
        disagree_parts.append(tagged)
    if not tfidf_wins.empty:
        tagged = _output_frame(tfidf_wins)
        tagged["who_wins"] = "tfidf"
        disagree_parts.append(tagged)
    if disagree_parts:
        disagreements = pd.concat(disagree_parts, ignore_index=True)
    else:
        disagreements = pd.DataFrame(columns=ERROR_COLUMNS + ["who_wins"])
        logger.warning("No disagreements to sample.")

    samples["tfidf_lr"].to_csv(C.ERRORS_DIR / "tfidf_lr_errors.csv", index=False)
    samples["sbert_lr"].to_csv(C.ERRORS_DIR / "sbert_lr_errors.csv", index=False)
    disagreements.to_csv(C.ERRORS_DIR / "disagreements.csv", index=False)
    summary = write_summary(samples["tfidf_lr"], samples["sbert_lr"], disagreements)
    summary_path = C.ERRORS_DIR / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    plot_reason_counts(samples["tfidf_lr"], samples["sbert_lr"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample and tag claim-verification errors.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    run()


if __name__ == "__main__":
    main()
