from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
METADATA_DIR = ROOT_DIR / "metadata"
OUTPUTS_AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
MANIFEST_PATH = METADATA_DIR / "file_manifest.csv"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"
SUMMARY_PATH = OUTPUTS_AUDIT_DIR / "00_inventory_summary.json"
TARGET_RELATIVE_PATHS = (
    Path("data/raw/answer.parquet"),
    Path("data/raw/question.parquet"),
)

MANIFEST_COLUMNS = [
    "file_name",
    "relative_path",
    "size_mb",
    "rows",
    "row_groups",
    "column_count",
    "columns",
    "time_column",
    "min_time",
    "max_time",
    "sha256",
    "schema_signature",
    "readable",
    "error_message",
]

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

TIME_NAME_KEYWORDS = (
    "time",
    "date",
    "datetime",
    "created",
    "updated",
    "published",
    "month",
    "day",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def find_parquet_files(raw_dir: Path) -> list[Path]:
    files: list[Path] = []
    for relative_path in TARGET_RELATIVE_PATHS:
        candidate = ROOT_DIR / relative_path
        if candidate.exists():
            files.append(candidate)
    return sorted(files)


def sha256_of_file(file_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def pick_time_column(schema: pa.Schema) -> str:
    typed_candidates: list[str] = []
    named_candidates: list[str] = []
    for field in schema:
        lowered = field.name.lower()
        is_temporal_type = pa.types.is_timestamp(field.type) or pa.types.is_date(field.type)
        has_time_name = any(keyword in lowered for keyword in TIME_NAME_KEYWORDS)
        if is_temporal_type:
            typed_candidates.append(field.name)
        elif has_time_name:
            named_candidates.append(field.name)
    if typed_candidates:
        return typed_candidates[0]
    if named_candidates:
        return named_candidates[0]
    return ""


def time_range_from_column(file_path: Path, column_name: str) -> tuple[str, str]:
    if not column_name:
        return "", ""
    try:
        table = pq.read_table(file_path, columns=[column_name])
        if table.num_rows == 0:
            return "", ""
        stats = pc.min_max(table.column(column_name))
        return normalize_scalar(stats["min"]), normalize_scalar(stats["max"])
    except Exception:
        return "", ""


def schema_signature(schema: pa.Schema) -> str:
    return "|".join(f"{field.name}:{field.type}" for field in schema)


def file_record(file_path: Path) -> dict[str, Any]:
    relative_path = file_path.relative_to(ROOT_DIR).as_posix()
    record: dict[str, Any] = {
        "file_name": file_path.name,
        "relative_path": relative_path,
        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 3),
        "rows": "",
        "row_groups": "",
        "column_count": "",
        "columns": "",
        "time_column": "",
        "min_time": "",
        "max_time": "",
        "sha256": "",
        "schema_signature": "",
        "readable": False,
        "error_message": "",
    }
    try:
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata
        schema = parquet_file.schema_arrow
        columns = [field.name for field in schema]
        chosen_time_column = pick_time_column(schema)
        min_time, max_time = time_range_from_column(file_path, chosen_time_column)

        record.update(
            {
                "rows": metadata.num_rows,
                "row_groups": metadata.num_row_groups,
                "column_count": len(columns),
                "columns": "|".join(columns),
                "time_column": chosen_time_column,
                "min_time": min_time,
                "max_time": max_time,
                "sha256": sha256_of_file(file_path),
                "schema_signature": schema_signature(schema),
                "readable": True,
            }
        )
    except Exception as exc:
        record["error_message"] = f"{type(exc).__name__}: {exc}"
        try:
            record["sha256"] = sha256_of_file(file_path)
        except Exception as hash_exc:
            record["error_message"] += f" | sha256_failed={type(hash_exc).__name__}: {hash_exc}"
    return record


def write_csv(file_path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_cleaning_log(rows: list[dict[str, str]]) -> None:
    file_exists = CLEANING_LOG_PATH.exists()
    is_empty = (not file_exists) or CLEANING_LOG_PATH.stat().st_size == 0
    with CLEANING_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANING_LOG_COLUMNS)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)


def infer_partition_hint(records: list[dict[str, Any]]) -> str:
    joined_names = " ".join(record["relative_path"].lower() for record in records)
    if re.search(r"(19|20)\d{2}[-_/]?(0[1-9]|1[0-2])", joined_names):
        return "likely_month_partition"
    if re.search(r"\b(batch|crawl|run|part)\b", joined_names):
        return "likely_batch_partition"
    if re.search(r"\b(dept|department|clinic|ward)\b", joined_names):
        return "likely_department_partition"
    return "entity_split_by_subject"


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    readable_records = [record for record in records if record["readable"]]
    unreadable_records = [record for record in records if not record["readable"]]
    schema_counts = Counter(record["schema_signature"] for record in readable_records)
    duplicate_groups = [
        {"sha256": sha256, "files": files}
        for sha256, files in _duplicate_sha_groups(records).items()
        if len(files) > 1
    ]
    return {
        "run_ts_utc": utc_now_iso(),
        "root_dir": str(ROOT_DIR),
        "raw_dir": str(RAW_DIR),
        "manifest_path": str(MANIFEST_PATH),
        "scope": "answer_question_only",
        "parquet_file_count": len(records),
        "readable_file_count": len(readable_records),
        "unreadable_file_count": len(unreadable_records),
        "distinct_schema_count": len(schema_counts),
        "all_schemas_consistent": len(schema_counts) <= 1,
        "duplicate_sha256_groups": duplicate_groups,
        "partition_hint": infer_partition_hint(records),
        "files": records,
    }


def _duplicate_sha_groups(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for record in records:
        sha256 = record["sha256"]
        if not sha256:
            continue
        groups.setdefault(sha256, []).append(record["relative_path"])
    return groups


def cleaning_log_entries(summary: dict[str, Any]) -> list[dict[str, str]]:
    timestamp = summary["run_ts_utc"]
    entries = [
        {
            "run_ts_utc": timestamp,
            "step": "00_inventory",
            "action": "scan_raw_parquet_files",
            "status": "completed",
            "target_path": str(RAW_DIR),
            "details": (
                f"scanned={summary['parquet_file_count']}; "
                f"readable={summary['readable_file_count']}; "
                f"unreadable={summary['unreadable_file_count']}; "
                f"scope={summary['scope']}"
            ),
        },
        {
            "run_ts_utc": timestamp,
            "step": "00_inventory",
            "action": "write_file_manifest",
            "status": "completed",
            "target_path": str(MANIFEST_PATH),
            "details": f"rows_written={summary['parquet_file_count']}",
        },
        {
            "run_ts_utc": timestamp,
            "step": "00_inventory",
            "action": "write_inventory_summary",
            "status": "completed",
            "target_path": str(SUMMARY_PATH),
            "details": (
                f"schema_consistent={summary['all_schemas_consistent']}; "
                f"duplicate_groups={len(summary['duplicate_sha256_groups'])}; "
                f"partition_hint={summary['partition_hint']}"
            ),
        },
    ]
    return entries


def main() -> None:
    ensure_directories()
    parquet_files = find_parquet_files(RAW_DIR)
    records = [file_record(file_path) for file_path in parquet_files]
    write_csv(MANIFEST_PATH, records, MANIFEST_COLUMNS)
    summary = build_summary(records)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    append_cleaning_log(cleaning_log_entries(summary))

    print(f"Scanned {summary['parquet_file_count']} parquet files from {RAW_DIR}")
    print(f"Manifest written to {MANIFEST_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Unreadable files: {summary['unreadable_file_count']}")
    print(f"Distinct schemas: {summary['distinct_schema_count']}")
    print(f"Duplicate SHA256 groups: {len(summary['duplicate_sha256_groups'])}")


if __name__ == "__main__":
    main()
