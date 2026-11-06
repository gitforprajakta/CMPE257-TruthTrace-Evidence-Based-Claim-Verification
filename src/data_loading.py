"""Download the official AVeriTeC train and dev JSON files.

Fallback chain (first success wins):
1. Hugging Face repo chenxwh/AVeriTeC (model repo, then dataset repo).
2. Hugging Face datasets pminervini/averitec.
3. Raw JSON from the GitHub repo MichSchli/AVeriTeC.

The official dev split is saved as dev.json. Stage 1 uses it as the test set
because the official test labels are not public.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from . import config as C
from .utils import get_logger, set_seed

logger = get_logger(__name__)

TRAIN_NAME = "train.json"
DEV_NAME = "dev.json"
SOURCE_NAME = "source.txt"

GITHUB_TRAIN = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/train.json"
GITHUB_DEV = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/dev.json"


def _destination(name: str) -> Path:
    return C.RAW_DIR / name


def raw_files_ready() -> bool:
    """Return True when train, dev, and the source marker are already on disk."""
    return all(_destination(name).exists() for name in (TRAIN_NAME, DEV_NAME, SOURCE_NAME))


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:80]
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com")


def load_record_list(path: Path) -> list:
    """Load a JSON list of claim records, accepting a few wrapper shapes."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("data", "examples", "claims", "instances"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            values = list(data.values())
            if values and all(isinstance(v, dict) for v in values):
                data = values
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError(f"{path} is not a non-empty list of JSON objects")
    if "claim" not in data[0]:
        raise ValueError(f"{path} records have no 'claim' field (keys={sorted(data[0])[:12]})")
    return data


def _publish(train_tmp: Path, dev_tmp: Path, source: str) -> None:
    """Validate a candidate download and move it into the raw directory."""
    if _looks_like_lfs_pointer(train_tmp) or _looks_like_lfs_pointer(dev_tmp):
        raise ValueError("download looks like a Git LFS pointer, not JSON")
    train_records = load_record_list(train_tmp)
    dev_records = load_record_list(dev_tmp)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(train_tmp, _destination(TRAIN_NAME))
    shutil.copyfile(dev_tmp, _destination(DEV_NAME))
    _destination(SOURCE_NAME).write_text(source + "\n", encoding="utf-8")
    logger.info(
        "Saved %s (%d records) and %s (%d records) from source %s",
        TRAIN_NAME, len(train_records), DEV_NAME, len(dev_records), source,
    )


def _try_chenxwh(workdir: Path) -> bool:
    """Download data/train.json and data/dev.json from chenxwh/AVeriTeC."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning("huggingface_hub is not installed; skipping chenxwh_hf")
        return False

    attempts: list[tuple[str, Optional[str]]] = [("default", None), ("dataset", "dataset")]
    for label, repo_type in attempts:
        try:
            kwargs = {"repo_id": "chenxwh/AVeriTeC"}
            if repo_type is not None:
                kwargs["repo_type"] = repo_type
            logger.info("Trying chenxwh/AVeriTeC via hf_hub_download (%s repo type)", label)
            train_cached = hf_hub_download(filename="data/train.json", **kwargs)
            dev_cached = hf_hub_download(filename="data/dev.json", **kwargs)
            train_tmp = workdir / "chenxwh_train.json"
            dev_tmp = workdir / "chenxwh_dev.json"
            shutil.copyfile(train_cached, train_tmp)
            shutil.copyfile(dev_cached, dev_tmp)
            _publish(train_tmp, dev_tmp, "chenxwh_hf")
            return True
        except Exception as exc:
            logger.warning("chenxwh_hf (%s) failed: %s", label, exc)
    return False


def _dataset_split(dataset, preferred: tuple[str, ...]):
    names = list(dataset.keys())
    for name in preferred:
        if name in dataset:
            return dataset[name], name
    raise KeyError(f"none of {preferred} in splits {names}")


def _try_pminervini(workdir: Path) -> bool:
    """Export pminervini/averitec to train.json and dev.json."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("datasets is not installed; skipping pminervini_hf")
        return False
    try:
        logger.info("Trying datasets.load_dataset('pminervini/averitec')")
        dataset = load_dataset("pminervini/averitec")
        train_split, train_name = _dataset_split(dataset, ("train",))
        dev_split, dev_name = _dataset_split(dataset, ("dev", "validation", "valid"))
        logger.info("pminervini/averitec splits used: train=%s dev=%s", train_name, dev_name)
        train_tmp = workdir / "pminervini_train.json"
        dev_tmp = workdir / "pminervini_dev.json"
        train_split.to_json(train_tmp)
        dev_split.to_json(dev_tmp)
        _normalize_json_or_jsonl(train_tmp)
        _normalize_json_or_jsonl(dev_tmp)
        _publish(train_tmp, dev_tmp, "pminervini_hf")
        return True
    except Exception as exc:
        logger.warning("pminervini_hf failed: %s", exc)
        return False


def _normalize_json_or_jsonl(path: Path) -> None:
    """Rewrite JSONL (datasets `.to_json`) as a JSON list when needed."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    if text[0] == "[":
        return
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def _download_url(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "TruthTrace/1.0 (CMPE257 research)"})
    with urllib.request.urlopen(request, timeout=180) as response, open(dest, "wb") as handle:
        shutil.copyfileobj(response, handle)


def _try_github(workdir: Path) -> bool:
    """Download train.json and dev.json from MichSchli/AVeriTeC."""
    try:
        logger.info("Trying GitHub raw files from MichSchli/AVeriTeC")
        train_tmp = workdir / "github_train.json"
        dev_tmp = workdir / "github_dev.json"
        _download_url(GITHUB_TRAIN, train_tmp)
        _download_url(GITHUB_DEV, dev_tmp)
        _publish(train_tmp, dev_tmp, "github")
        return True
    except Exception as exc:
        logger.warning("github source failed: %s", exc)
        return False


def download(force: bool = False) -> str:
    """Ensure raw train/dev JSON and source.txt exist. Return the source name."""
    set_seed()
    C.ensure_dirs()
    if raw_files_ready() and not force:
        source = _destination(SOURCE_NAME).read_text(encoding="utf-8").strip()
        logger.info("Raw AVeriTeC files already present (source=%s). Use --force to redownload.", source)
        return source

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for attempt in (_try_chenxwh, _try_pminervini, _try_github):
            if attempt(workdir):
                return _destination(SOURCE_NAME).read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "Could not download AVeriTeC from chenxwh/AVeriTeC, pminervini/averitec, or MichSchli/AVeriTeC"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AVeriTeC train and dev JSON.")
    parser.add_argument("--force", action="store_true", help="Redownload even if raw files exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = download(force=args.force)
    logger.info("Data source: %s", source)


if __name__ == "__main__":
    main()
