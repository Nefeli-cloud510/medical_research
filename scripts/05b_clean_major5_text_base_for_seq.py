from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

INPUT_PATH = ANALYSIS_DIR / "major5_text_cleaning_prep.parquet"
OUTPUT_PATH = ANALYSIS_DIR / "major5_text_cleaning_prep_seq_clean.parquet"
SUMMARY_PATH = AUDIT_DIR / "15_major5_text_cleaning_seq_clean_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

DROP_COLUMNS = [
    "gender",
    "age",
    "address",
    "is_same_doctor_multi_answer_in_question",
    "is_doctor_first_answer_in_question",
    "is_doctor_followup_answer_in_question",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def append_cleaning_log(rows: list[dict[str, str]]) -> None:
    file_exists = CLEANING_LOG_PATH.exists()
    is_empty = (not file_exists) or CLEANING_LOG_PATH.stat().st_size == 0
    with CLEANING_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANING_LOG_COLUMNS)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_directories()
    df = pd.read_parquet(INPUT_PATH).copy()

    original_rows = int(len(df))
    original_columns = list(df.columns)

    time_seq_equals_observed = bool(df["time_seq"].equals(df["observed_seq_by_time"]))

    existing_drop_columns = [column for column in DROP_COLUMNS if column in df.columns]
    df = df.drop(columns=existing_drop_columns)

    # For all downstream text work, use within-question chronological order as the primary view.
    df = df.sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "major5_text_cleaning_seq_clean",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "row_counts": {
            "original_rows": original_rows,
            "output_rows": int(len(df)),
        },
        "dropped_columns": existing_drop_columns,
        "ordering_explanation": {
            "recommended_field_for_within_question_time_order": "time_seq",
            "sort_order_applied": ["question_id", "time_seq", "answer_create_time", "seq"],
            "time_seq_definition": "按同一 question_id 内的 answer_create_time 从早到晚重新排序后的位次。",
            "observed_seq_by_time_definition": "历史审计别名字段，当前实现中与 time_seq 完全相同。",
            "time_seq_equals_observed_seq_by_time": time_seq_equals_observed,
            "recommendation": "后续所有题内先后顺序、前序回答集合、文本模板构造统一看 time_seq，不再以 observed_seq_by_time 作为主字段。",
        },
        "column_audit": {
            "original_column_count": len(original_columns),
            "output_column_count": len(df.columns),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05b_clean_major5_text_base_for_seq",
                "action": "drop_unused_columns_and_resort_by_time_seq",
                "status": "completed",
                "target_path": str(OUTPUT_PATH),
                "details": (
                    f"input={INPUT_PATH.name}; rows={original_rows}; "
                    f"dropped_columns={'|'.join(existing_drop_columns)}; "
                    f"use_field_for_order=time_seq"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05b_clean_major5_text_base_for_seq",
                "action": "write_seq_clean_summary",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"time_seq_equals_observed={time_seq_equals_observed}; "
                    f"sort_order=question_id|time_seq|answer_create_time|seq"
                ),
            },
        ]
    )

    print(f"Seq-clean text base written to {OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
