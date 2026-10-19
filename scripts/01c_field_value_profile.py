from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
METADATA_DIR = ROOT_DIR / "metadata"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"
SUMMARY_PATH = AUDIT_DIR / "03_field_value_profile_summary.json"
PROFILE_PATH = TABLES_DIR / "03_field_value_profile.csv"
COUNTS_PATH = TABLES_DIR / "03_field_value_counts.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

PROFILE_CONFIG: dict[str, dict[str, str]] = {
    "answer.parquet": {
        "question_id": "top_100",
        "user_id": "top_100",
        "tech_title": "all",
        "dept": "top_100",
        "hospital": "top_100",
        "answer": "text_summary",
        "create_time": "top_100",
        "thumb": "top_100",
        "help_status": "all",
        "seq": "all",
    },
    "question.parquet": {
        "question_id": "all",
        "create_time": "top_100",
        "gender": "all",
        "age": "all",
        "address": "top_100",
        "question_desc": "text_summary",
        "tags": "top_100",
        "bounty": "all",
        "views": "top_100",
        "ans_count": "all",
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def append_cleaning_log(rows: list[dict[str, str]]) -> None:
    file_exists = CLEANING_LOG_PATH.exists()
    is_empty = (not file_exists) or CLEANING_LOG_PATH.stat().st_size == 0
    with CLEANING_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANING_LOG_COLUMNS)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)


def clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def clean_string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def profile_series(
    file_name: str,
    field_name: str,
    series: pd.Series,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_count = int(series.shape[0])
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        cleaned = series.map(clean_string_value)
        missing_mask = cleaned.eq("")
        non_missing = cleaned.loc[~missing_mask]
    else:
        missing_mask = series.isna()
        non_missing = series.loc[~missing_mask]

    missing_count = int(missing_mask.sum())
    non_missing_count = int(non_missing.shape[0])
    distinct_count = int(non_missing.nunique(dropna=True))
    repeated_count = int(non_missing_count - distinct_count)

    profile_row: dict[str, Any] = {
        "file_name": file_name,
        "field_name": field_name,
        "profile_mode": mode,
        "total_count": total_count,
        "non_missing_count": non_missing_count,
        "missing_count": missing_count,
        "distinct_count": distinct_count,
        "repeated_count": repeated_count,
        "duplicate_rate_non_missing": round(repeated_count / non_missing_count, 6) if non_missing_count else 0.0,
        "min_value": "",
        "max_value": "",
        "mean_value": "",
        "median_value": "",
        "note": "",
    }

    if pd.api.types.is_numeric_dtype(non_missing.dtype) and non_missing_count:
        numeric = pd.to_numeric(non_missing, errors="coerce").dropna()
        if not numeric.empty:
            profile_row["min_value"] = clean_scalar(numeric.min())
            profile_row["max_value"] = clean_scalar(numeric.max())
            profile_row["mean_value"] = round(float(numeric.mean()), 6)
            profile_row["median_value"] = round(float(numeric.median()), 6)
    elif pd.api.types.is_datetime64_any_dtype(non_missing.dtype) and non_missing_count:
        profile_row["min_value"] = clean_scalar(non_missing.min())
        profile_row["max_value"] = clean_scalar(non_missing.max())

    count_rows: list[dict[str, Any]] = []
    if mode == "text_summary":
        text_lengths = non_missing.map(lambda x: len(str(x)))
        if not text_lengths.empty:
            profile_row["min_value"] = int(text_lengths.min())
            profile_row["max_value"] = int(text_lengths.max())
            profile_row["mean_value"] = round(float(text_lengths.mean()), 6)
            profile_row["median_value"] = round(float(text_lengths.median()), 6)
        profile_row["note"] = "high_cardinality_text_summary_only"
        return profile_row, count_rows

    value_counts = non_missing.value_counts(dropna=False)
    if mode == "top_100":
        selected_counts = value_counts.head(100)
        profile_row["note"] = "top_100_value_counts"
    else:
        selected_counts = value_counts
        profile_row["note"] = "all_value_counts"

    for rank, (value, count) in enumerate(selected_counts.items(), start=1):
        count_rows.append(
            {
                "file_name": file_name,
                "field_name": field_name,
                "rank": rank,
                "value": clean_scalar(value),
                "count": int(count),
                "share_in_non_missing": round(float(count) / non_missing_count, 6) if non_missing_count else 0.0,
            }
        )

    return profile_row, count_rows


def main() -> None:
    ensure_directories()
    profile_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []

    for file_name, field_modes in PROFILE_CONFIG.items():
        file_path = RAW_DIR / file_name
        for field_name, mode in field_modes.items():
            series = pd.read_parquet(file_path, columns=[field_name])[field_name]
            profile_row, field_count_rows = profile_series(file_name, field_name, series, mode)
            profile_rows.append(profile_row)
            count_rows.extend(field_count_rows)

    profile_df = pd.DataFrame(profile_rows)
    count_df = pd.DataFrame(count_rows)

    profile_df.to_csv(PROFILE_PATH, index=False, encoding="utf-8-sig")
    count_df.to_csv(COUNTS_PATH, index=False, encoding="utf-8-sig")

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only",
        "profile_table": str(PROFILE_PATH),
        "count_table": str(COUNTS_PATH),
        "field_count": int(profile_df.shape[0]),
        "count_row_count": int(count_df.shape[0]),
        "text_summary_fields": sorted(profile_df.loc[profile_df["profile_mode"] == "text_summary", "field_name"].tolist()),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "01c_field_value_profile",
                "action": "profile_field_values_and_counts",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": f"fields={summary['field_count']}; count_rows={summary['count_row_count']}; scope={summary['scope']}",
            }
        ]
    )

    print(f"Field profile written to {PROFILE_PATH}")
    print(f"Value counts written to {COUNTS_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
