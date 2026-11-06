"""Exploratory figures and a per-class summary table for the processed splits."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config as C
from .utils import get_logger, set_seed

logger = get_logger(__name__)

FIGURE_NAMES = [
    "eda_class_distribution.png",
    "eda_claim_length.png",
    "eda_evidence_length.png",
    "eda_missing_evidence_by_class.png",
    "eda_n_qa_by_class.png",
    "eda_nee_phrase_check.png",
]
SUMMARY_NAME = "eda_summary.csv"
SHORTCUT_PHRASES = ("no answer could be found", "no answer")


def _short(label: str) -> str:
    return C.SHORT_LABELS.get(label, label)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not C.TRAIN_PARQUET.exists() or not C.TEST_PARQUET.exists():
        raise FileNotFoundError("Processed parquet files are missing. Run python -m src.preprocess first.")
    train = pd.read_parquet(C.TRAIN_PARQUET)
    test = pd.read_parquet(C.TEST_PARQUET)
    return train, test


def combined(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [train.assign(split="train"), test.assign(split="test")],
        ignore_index=True,
    )


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, name: str) -> None:
    path = C.FIGURES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def plot_class_distribution(frame: pd.DataFrame) -> None:
    """Grouped train vs test counts with the within-split percentage on each bar."""
    rows = []
    for split, part in frame.groupby("split"):
        total = len(part)
        for label in C.LABELS:
            count = int((part["label"] == label).sum())
            rows.append({
                "split": split,
                "label": _short(label),
                "count": count,
                "share": (count / total) if total else 0.0,
            })
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=plot_df, x="label", y="count", hue="split", hue_order=["train", "test"],
        ax=ax, order=[_short(l) for l in C.LABELS],
    )
    for container, split in zip(ax.containers, ["train", "test"]):
        shares = plot_df.loc[plot_df["split"] == split, "share"].tolist()
        ax.bar_label(container, labels=[f"{100 * share:.1f}%" for share in shares], fontsize=8, padding=2)
    ax.set_xlabel("Label")
    ax.set_ylabel("Claims")
    ax.set_title("Class distribution")
    fig.tight_layout()
    _save(fig, "eda_class_distribution.png")


def plot_length(frame: pd.DataFrame, column: str, name: str, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        data=frame, x=column, hue="split", hue_order=["train", "test"],
        element="step", stat="count", bins=30, ax=ax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Claims")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, name)


def plot_missing_evidence(frame: pd.DataFrame) -> None:
    rows = []
    for (split, label), part in frame.groupby(["split", "label"]):
        rows.append({
            "split": split,
            "label": _short(label),
            "missing_share": float((part["has_evidence"] == 0).mean()) if len(part) else 0.0,
        })
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=plot_df, x="label", y="missing_share", hue="split", hue_order=["train", "test"],
        ax=ax, order=[_short(l) for l in C.LABELS],
    )
    ax.set_xlabel("Label")
    ax.set_ylabel("Share with no evidence")
    ax.set_ylim(0, 1)
    ax.set_title("Missing evidence by class")
    fig.tight_layout()
    _save(fig, "eda_missing_evidence_by_class.png")


def plot_n_qa(frame: pd.DataFrame) -> None:
    plot_df = frame.copy()
    plot_df["label_short"] = plot_df["label"].map(_short)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=plot_df, x="label_short", y="n_qa", hue="split", hue_order=["train", "test"],
        ax=ax, order=[_short(l) for l in C.LABELS],
    )
    ax.set_xlabel("Label")
    ax.set_ylabel("QA pairs")
    ax.set_title("Evidence questions per class")
    fig.tight_layout()
    _save(fig, "eda_n_qa_by_class.png")


def shortcut_mask(evidence: pd.Series) -> pd.Series:
    text = evidence.fillna("").str.casefold()
    mask = pd.Series(False, index=evidence.index)
    for phrase in SHORTCUT_PHRASES:
        mask = mask | text.str.contains(phrase, regex=False)
    return mask


def plot_nee_phrases(frame: pd.DataFrame) -> None:
    """Share of evidence texts that contain the annotation-shortcut phrases."""
    marked = frame.copy()
    marked["has_phrase"] = shortcut_mask(marked["evidence_text"])
    rows = []
    for (split, label), part in marked.groupby(["split", "label"]):
        rows.append({
            "split": split,
            "label": _short(label),
            "phrase_share": float(part["has_phrase"].mean()) if len(part) else 0.0,
        })
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=plot_df, x="label", y="phrase_share", hue="split", hue_order=["train", "test"],
        ax=ax, order=[_short(l) for l in C.LABELS],
    )
    ax.set_xlabel("Label")
    ax.set_ylabel("Share containing a shortcut phrase")
    ax.set_ylim(0, 1)
    ax.set_title("Evidence shortcut phrases by class")
    fig.tight_layout()
    _save(fig, "eda_nee_phrase_check.png")


def build_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per split and label: count, share, lengths, missing-evidence share."""
    rows = []
    for split, part in (("train", train), ("test", test)):
        total = len(part)
        for label in C.LABELS:
            subset = part.loc[part["label"] == label]
            count = int(len(subset))
            rows.append({
                "split": split,
                "label": label,
                "count": count,
                "share": float(count / total) if total else 0.0,
                "mean_claim_len": float(subset["claim_len"].mean()) if count else 0.0,
                "mean_evidence_len": float(subset["evidence_len"].mean()) if count else 0.0,
                "missing_evidence_share": float((subset["has_evidence"] == 0).mean()) if count else 0.0,
            })
    return pd.DataFrame(rows)


def outputs_ready() -> bool:
    figures = all((C.FIGURES_DIR / name).exists() for name in FIGURE_NAMES)
    return figures and (C.TABLES_DIR / SUMMARY_NAME).exists()


def run(force: bool = False) -> None:
    set_seed()
    C.ensure_dirs()
    _style()
    if outputs_ready() and not force:
        logger.info("EDA outputs already present. Use --force to recompute.")
        return
    train, test = load_splits()
    frame = combined(train, test)
    plot_class_distribution(frame)
    plot_length(frame, "claim_len", "eda_claim_length.png", "Claim length", "Words in the claim")
    plot_length(
        frame, "evidence_len", "eda_evidence_length.png", "Evidence length",
        "Words in evidence (0 = no evidence)",
    )
    plot_missing_evidence(frame)
    plot_n_qa(frame)
    plot_nee_phrases(frame)
    summary = build_summary(train, test)
    path = C.TABLES_DIR / SUMMARY_NAME
    summary.to_csv(path, index=False)
    logger.info("Wrote %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write EDA figures and eda_summary.csv.")
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
