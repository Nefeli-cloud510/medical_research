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

INPUT_PATH = ANALYSIS_DIR / "focal_answer_panel_base.parquet"
OUTPUT_PATH = ANALYSIS_DIR / "focal_answer_panel_base_time_valid.parquet"
SUMMARY_PATH = AUDIT_DIR / "06_drop_invalid_time_rows_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
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
    df = pd.read_parquet(INPUT_PATH)
    original_rows = int(len(df))
    invalid_rows = int((~df["is_answer_time_valid"]).sum())
    filtered_df = df.loc[df["is_answer_time_valid"]].copy()
    filtered_rows = int(len(filtered_df))

    filtered_df.to_parquet(OUTPUT_PATH, index=False)

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "original_rows": original_rows,
        "dropped_invalid_time_rows": invalid_rows,
        "filtered_rows": filtered_rows,
        "drop_rate": round(invalid_rows / original_rows, 6) if original_rows else 0.0,
        "filter_rule": "keep rows where is_answer_time_valid == True",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "03d_drop_invalid_time_rows",
                "action": "drop_invalid_time_rows_and_save_new_version",
                "status": "completed",
                "target_path": str(OUTPUT_PATH),
                "details": (
                    f"original_rows={original_rows}; "
                    f"dropped_invalid_time_rows={invalid_rows}; "
                    f"filtered_rows={filtered_rows}"
                ),
            }
        ]
    )

    print(f"Filtered table written to {OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
