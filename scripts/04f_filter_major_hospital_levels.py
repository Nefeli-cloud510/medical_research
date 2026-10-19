from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

INPUT_PATH = (
    ANALYSIS_DIR
    / "focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_filled_round2.parquet"
)
OUTPUT_PATH = (
    ANALYSIS_DIR
    / "focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_major5.parquet"
)
SUMMARY_PATH = AUDIT_DIR / "12_filter_major_hospital_levels_summary.json"
DISTRIBUTION_PATH = TABLES_DIR / "12_hospitallevel_distribution_after_major5_filter.csv"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

KEEP_LEVELS = [
    "一级甲等",
    "二级乙等",
    "三级乙等",
    "三级甲等",
    "二级甲等",
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


def clean_string(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def build_distribution(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = int(len(df))
    distinct_hospital_df = (
        df.assign(hospital=clean_string(df["hospital"]))
        .loc[lambda frame: frame["hospital"] != "", ["hospital", "hospitallevel_final"]]
        .drop_duplicates(subset=["hospital"])
        .reset_index(drop=True)
    )
    distinct_hospitals = int(len(distinct_hospital_df))

    distribution_df = (
        df["hospitallevel_final"]
        .value_counts(dropna=False)
        .rename_axis("hospitallevel")
        .reset_index(name="row_count")
    )
    distribution_df["row_share"] = (distribution_df["row_count"] / total_rows).round(6) if total_rows else 0.0

    distinct_distribution_df = (
        distinct_hospital_df["hospitallevel_final"]
        .value_counts(dropna=False)
        .rename_axis("hospitallevel")
        .reset_index(name="distinct_hospital_count")
    )
    distinct_distribution_df["distinct_hospital_share"] = (
        distinct_distribution_df["distinct_hospital_count"] / distinct_hospitals
    ).round(6) if distinct_hospitals else 0.0

    return distribution_df.merge(distinct_distribution_df, on="hospitallevel", how="left")


def main() -> None:
    ensure_directories()
    df = pd.read_parquet(INPUT_PATH).copy()
    df["hospitallevel_final"] = clean_string(df["hospitallevel_final"])

    original_rows = int(len(df))
    original_distribution = (
        df["hospitallevel_final"].replace("", pd.NA).fillna("UNMATCHED").value_counts(dropna=False).to_dict()
    )

    filtered_df = df.loc[df["hospitallevel_final"].isin(KEEP_LEVELS)].copy()
    filtered_rows = int(len(filtered_df))
    dropped_rows = original_rows - filtered_rows

    filtered_df.to_parquet(OUTPUT_PATH, index=False)

    distribution_df = build_distribution(filtered_df)
    distribution_df.to_csv(DISTRIBUTION_PATH, index=False, encoding="utf-8-sig")

    kept_distribution = distribution_df.set_index("hospitallevel")["row_count"].to_dict()
    dropped_distribution = {
        str(level): int(count)
        for level, count in original_distribution.items()
        if str(level) not in KEEP_LEVELS
    }

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only_core_physician_titles_major5_hospital_levels",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "filter_rule": {
            "keep_levels": KEEP_LEVELS,
            "drop_other_hospital_levels": True,
        },
        "row_counts": {
            "original_rows": original_rows,
            "filtered_rows": filtered_rows,
            "dropped_rows": dropped_rows,
            "keep_rate": round(filtered_rows / original_rows, 6) if original_rows else 0.0,
            "drop_rate": round(dropped_rows / original_rows, 6) if original_rows else 0.0,
        },
        "distribution_before": original_distribution,
        "distribution_after": kept_distribution,
        "dropped_distribution": dropped_distribution,
        "top_distribution_rows_after_filter": distribution_df.to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "04f_filter_major_hospital_levels",
                "action": "keep_only_major5_hospital_levels_and_save_new_version",
                "status": "completed",
                "target_path": str(OUTPUT_PATH),
                "details": (
                    f"original_rows={original_rows}; "
                    f"filtered_rows={filtered_rows}; "
                    f"dropped_rows={dropped_rows}; "
                    f"keep_levels={'|'.join(KEEP_LEVELS)}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "04f_filter_major_hospital_levels",
                "action": "write_major5_distribution_outputs",
                "status": "completed",
                "target_path": str(DISTRIBUTION_PATH),
                "details": f"kept_levels={len(KEEP_LEVELS)}; distinct_levels_after_filter={len(distribution_df)}",
            },
        ]
    )

    print(f"Filtered table written to {OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Distribution written to {DISTRIBUTION_PATH}")


if __name__ == "__main__":
    main()
