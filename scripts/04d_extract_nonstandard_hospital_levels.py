from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
METADATA_DIR = ROOT_DIR / "metadata"

INPUT_PATH = ANALYSIS_DIR / "focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_filled.parquet"
OUTPUT_PATH = TABLES_DIR / "10_nonstandard_hospital_levels_for_manual_search.csv"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

TARGET_LEVELS = [
    "一级医院",
    "二级医院",
    "三级医院",
    "其他",
    "对外综合",
    "对外专科",
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
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


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

    keep_cols = [
        "hospital",
        "hospitallevel_final",
        "hospitallevel_fill_method",
        "hospitallevel_source",
        "hospitallevel_note",
    ]
    extracted = (
        df.loc[df["hospitallevel_final"].isin(TARGET_LEVELS), keep_cols]
        .drop_duplicates(subset=["hospital"])
        .sort_values(["hospitallevel_final", "hospital"], kind="stable")
        .reset_index(drop=True)
    )
    extracted.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    append_cleaning_log(
        [
            {
                "run_ts_utc": utc_now_iso(),
                "step": "04d_extract_nonstandard_hospital_levels",
                "action": "extract_nonstandard_hospital_levels_for_manual_search",
                "status": "completed",
                "target_path": str(OUTPUT_PATH),
                "details": (
                    f"rows={len(extracted)}; "
                    f"target_levels={'|'.join(TARGET_LEVELS)}"
                ),
            }
        ]
    )

    print(f"CSV written to {OUTPUT_PATH}")
    print(f"rows={len(extracted)}")


if __name__ == "__main__":
    main()
