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

INPUT_PATH = ANALYSIS_DIR / "focal_answer_panel_base_time_valid.parquet"
OUTPUT_PATH = ANALYSIS_DIR / "focal_answer_panel_base_time_valid_core_physician_titles.parquet"
SUMMARY_PATH = AUDIT_DIR / "07_filter_core_physician_titles_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

KEEP_TITLES = [
    "副主任医师",
    "主治医师",
    "主任医师",
    "住院医师",
    "医师",
]

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
    original_title_counts = df["tech_title"].value_counts(dropna=False).to_dict()
    has_delimiter_rows = int(
        df["tech_title"].astype("string").str.contains(r"[|,，、;/；]", regex=True, na=False).sum()
    )

    filtered_df = df.loc[df["tech_title"].isin(KEEP_TITLES)].copy()
    filtered_rows = int(len(filtered_df))
    dropped_rows = original_rows - filtered_rows
    kept_title_counts = filtered_df["tech_title"].value_counts().to_dict()

    filtered_df.to_parquet(OUTPUT_PATH, index=False)

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "validation": {
            "tech_title_is_single_value_per_row": has_delimiter_rows == 0,
            "rows_with_delimiter_in_tech_title": has_delimiter_rows,
        },
        "filter_rule": {
            "keep_titles": KEEP_TITLES,
            "drop_other_titles": True,
        },
        "row_counts": {
            "original_rows": original_rows,
            "filtered_rows": filtered_rows,
            "dropped_rows": dropped_rows,
            "drop_rate": round(dropped_rows / original_rows, 6) if original_rows else 0.0,
            "keep_rate": round(filtered_rows / original_rows, 6) if original_rows else 0.0,
        },
        "title_counts_before": original_title_counts,
        "title_counts_after": kept_title_counts,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "03e_filter_core_physician_titles",
                "action": "keep_only_five_core_physician_titles_and_save_new_version",
                "status": "completed",
                "target_path": str(OUTPUT_PATH),
                "details": (
                    f"original_rows={original_rows}; "
                    f"filtered_rows={filtered_rows}; "
                    f"dropped_rows={dropped_rows}; "
                    f"keep_titles={'|'.join(KEEP_TITLES)}"
                ),
            }
        ]
    )

    print(f"Filtered table written to {OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
