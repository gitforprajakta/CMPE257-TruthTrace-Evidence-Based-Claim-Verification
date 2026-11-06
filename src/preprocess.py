"""Turn raw AVeriTeC JSON into the processed train/test contract.

The official dev split is written to test.parquet. The justification field is
never copied into any output column.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from typing import Any

import pandas as pd

from . import config as C
from .data_loading import DEV_NAME, TRAIN_NAME, load_record_list
from .utils import get_logger, set_seed, write_json

logger = get_logger(__name__)

PROCESSED_COLUMNS = [
    "claim_id",
    "claim",
    "label",
    "label_id",
    "evidence_text",
    "evidence_pairs",
    "has_evidence",
    "n_qa",
    "claim_len",
    "evidence_len",
    "speaker",
    "claim_date",
    "combined_text",
]

_LABEL_MAP = {
    "supported": "Supported",
    "refuted": "Refuted",
    "not enough evidence": "Not Enough Evidence",
    "not enough information": "Not Enough Evidence",
    "nee": "Not Enough Evidence",
    "conflicting evidence/cherrypicking": "Conflicting Evidence/Cherrypicking",
    "conflicting evidence / cherrypicking": "Conflicting Evidence/Cherrypicking",
    "conflicting evidence": "Conflicting Evidence/Cherrypicking",
    "conflicting": "Conflicting Evidence/Cherrypicking",
    "cherrypicking": "Conflicting Evidence/Cherrypicking",
}

_WS = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    """NFKC-normalize, collapse whitespace, and strip. None becomes ''."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return _WS.sub(" ", text).strip()


def normalize_label(value: Any) -> str:
    """Map a raw label onto config.LABELS.

    Blank labels raise ValueError('empty label'). Any other string that is not
    a known variant raises ValueError listing that label.
    """
    key = normalize_text(value).casefold()
    if not key:
        raise ValueError("empty label")
    if key not in _LABEL_MAP:
        raise ValueError(f"Unmappable label: {value!r}")
    return _LABEL_MAP[key]


def _as_list(value: Any) -> list:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                return []
        else:
            return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def build_evidence_pairs(record: dict) -> list[str]:
    """Build 'Q: ... A: ...' strings from the first three answers of each question.

    Boolean answers append a non-empty boolean_explanation. The justification
    field is never read.
    """
    pairs: list[str] = []
    for question in _as_list(record.get("questions")):
        if not isinstance(question, dict):
            continue
        question_text = normalize_text(question.get("question") or question.get("question_text"))
        answers = _as_list(question.get("answers"))[:3]
        for answer in answers:
            if isinstance(answer, str):
                answer_text = normalize_text(answer)
                answer_type = ""
                explanation = ""
            elif isinstance(answer, dict):
                answer_text = normalize_text(
                    answer.get("answer") if answer.get("answer") is not None else answer.get("text")
                )
                answer_type = normalize_text(answer.get("answer_type"))
                explanation = normalize_text(answer.get("boolean_explanation"))
            else:
                continue
            if not answer_text:
                continue
            pair = f"Q: {question_text} A: {answer_text}"
            if answer_type.casefold() == "boolean" and explanation:
                pair += f" Explanation: {explanation}"
            pairs.append(pair)
    return pairs


def _claim_key(claim: str) -> str:
    return claim.casefold()


def _row_from_record(record: dict, claim: str, label: str) -> dict:
    pairs = build_evidence_pairs(record)
    if pairs:
        evidence_text = " [SEP] ".join(pairs)
        has_evidence = 1
        n_qa = len(pairs)
        evidence_len = _word_count(evidence_text)
        evidence_pairs = pairs
    else:
        evidence_text = C.NO_EVIDENCE_TOKEN
        has_evidence = 0
        n_qa = 0
        evidence_len = 0
        evidence_pairs = [C.NO_EVIDENCE_TOKEN]
    speaker = normalize_text(record.get("speaker"))
    claim_date = normalize_text(record.get("claim_date"))
    return {
        "claim": claim,
        "label": label,
        "label_id": int(C.LABEL2ID[label]),
        "evidence_text": evidence_text,
        "evidence_pairs": evidence_pairs,
        "has_evidence": int(has_evidence),
        "n_qa": int(n_qa),
        "claim_len": int(_word_count(claim)),
        "evidence_len": int(evidence_len),
        "speaker": speaker,
        "claim_date": claim_date,
        "combined_text": f"{claim} [SEP] {evidence_text}",
    }


def _prepare_split(records: list, split_name: str) -> tuple[list[dict], int, int]:
    rows: list[dict] = []
    dropped_invalid = 0
    dropped_duplicates = 0
    seen: set[str] = set()
    unmappable: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            dropped_invalid += 1
            continue
        raw_label = record.get("label")
        if raw_label is None or normalize_text(raw_label) == "":
            dropped_invalid += 1
            continue
        try:
            label = normalize_label(raw_label)
        except ValueError:
            unmappable.append(str(raw_label))
            dropped_invalid += 1
            continue
        claim = normalize_text(record.get("claim"))
        if not claim:
            dropped_invalid += 1
            continue
        key = _claim_key(claim)
        if key in seen:
            dropped_duplicates += 1
            continue
        seen.add(key)
        rows.append(_row_from_record(record, claim, label))
    if unmappable:
        unique = sorted(set(unmappable))
        raise ValueError(
            f"Unmappable labels in {split_name}: {unique}. "
            f"Expected one of {list(C.LABELS)} or a documented variant."
        )
    return rows, dropped_invalid, dropped_duplicates


def _assign_ids(rows: list[dict]) -> pd.DataFrame:
    for index, row in enumerate(rows):
        row["claim_id"] = int(index)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=PROCESSED_COLUMNS)
    return frame[PROCESSED_COLUMNS]


def _class_counts(rows: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in C.LABELS}
    for row in rows:
        counts[row["label"]] += 1
    return counts


def _missing_share(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    missing = sum(1 for row in rows if row["has_evidence"] == 0)
    return float(missing / len(rows))


def preprocess_records(
    train_records: list,
    test_records: list,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean, deduplicate, and build the Section 5 frames plus the JSON report."""
    train_rows, train_invalid, train_dupes = _prepare_split(train_records, "train")
    test_rows, test_invalid, test_dupes = _prepare_split(test_records, "test")
    train_loaded = len(train_records)
    test_loaded = len(test_records)

    train_keys = {_claim_key(row["claim"]) for row in train_rows}
    kept_test: list[dict] = []
    removed_overlap = 0
    for row in test_rows:
        if _claim_key(row["claim"]) in train_keys:
            removed_overlap += 1
            continue
        kept_test.append(row)

    train_df = _assign_ids(train_rows)
    test_df = _assign_ids(kept_test)
    report = {
        "source": source,
        "train": {
            "loaded": int(train_loaded),
            "dropped_invalid": int(train_invalid),
            "dropped_duplicates": int(train_dupes),
            "final": int(len(train_df)),
            "missing_evidence_share": _missing_share(train_rows),
            "class_counts": _class_counts(train_rows),
        },
        "test": {
            "loaded": int(test_loaded),
            "dropped_invalid": int(test_invalid),
            "dropped_duplicates": int(test_dupes),
            "removed_train_overlap": int(removed_overlap),
            "final": int(len(test_df)),
            "missing_evidence_share": _missing_share(kept_test),
            "class_counts": _class_counts(kept_test),
        },
    }
    return train_df, test_df, report


def outputs_ready() -> bool:
    return C.TRAIN_PARQUET.exists() and C.TEST_PARQUET.exists() and C.PREPROCESS_REPORT.exists()


def read_source_name() -> str:
    path = C.RAW_DIR / "source.txt"
    if not path.exists():
        return "unknown"
    return path.read_text(encoding="utf-8").strip() or "unknown"


def run(force: bool = False) -> dict:
    """Write train.parquet, test.parquet, and preprocess_report.json."""
    set_seed()
    C.ensure_dirs()
    if outputs_ready() and not force:
        logger.info("Processed files already present. Use --force to recompute.")
        from .utils import read_json
        return read_json(C.PREPROCESS_REPORT)

    train_path = C.RAW_DIR / TRAIN_NAME
    dev_path = C.RAW_DIR / DEV_NAME
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            f"Missing raw files in {C.RAW_DIR}. Run python -m src.data_loading first."
        )
    source = read_source_name()
    train_records = load_record_list(train_path)
    dev_records = load_record_list(dev_path)
    if train_records:
        logger.info("Train record keys: %s", sorted(train_records[0].keys()))
    train_df, test_df, report = preprocess_records(train_records, dev_records, source)
    if set(train_df["label"]).isdisjoint(C.LABELS) or any(
        label not in set(train_df["label"]) for label in C.LABELS
    ):
        present = sorted(set(train_df["label"]))
        raise RuntimeError(f"Train split is missing one of the four labels. Present: {present}")
    overlap = set(train_df["claim"].str.casefold()) & set(test_df["claim"].str.casefold())
    if overlap:
        raise RuntimeError(f"Train/test claim overlap remains after preprocessing ({len(overlap)} claims)")

    train_df.to_parquet(C.TRAIN_PARQUET, index=False, engine="pyarrow")
    test_df.to_parquet(C.TEST_PARQUET, index=False, engine="pyarrow")
    write_json(report, C.PREPROCESS_REPORT)
    logger.info(
        "Wrote %s (%d rows) and %s (%d rows); source=%s",
        C.TRAIN_PARQUET.name, len(train_df), C.TEST_PARQUET.name, len(test_df), source,
    )
    logger.info("Train class counts: %s", report["train"]["class_counts"])
    logger.info("Test class counts: %s", report["test"]["class_counts"])
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess AVeriTeC train and dev into parquet.")
    parser.add_argument("--force", action="store_true", help="Recompute even if parquet files exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
