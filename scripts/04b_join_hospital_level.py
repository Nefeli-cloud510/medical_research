from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
METADATA_DIR = ROOT_DIR / "metadata"

INPUT_ANALYSIS_PATH = ANALYSIS_DIR / "focal_answer_panel_base_time_valid_core_physician_titles.parquet"
INPUT_HOSPITAL_PATH = RAW_DIR / "wy_hospital.parquet"
OUTPUT_ANALYSIS_PATH = ANALYSIS_DIR / "focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel.parquet"
SUMMARY_PATH = AUDIT_DIR / "08_hospitallevel_join_summary.json"
DISTRIBUTION_PATH = TABLES_DIR / "08_hospitallevel_distribution.csv"
UNMATCHED_PATH = TABLES_DIR / "08_unmatched_hospitals.csv"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

HOSPITAL_COL_CANDIDATES = ["hospital", "hospital_name", "hospitalName", "name", "医院", "医院名称"]
LEVEL_COL_CANDIDATES = ["hospitallevel", "hospital_level", "hospitalLevel", "level", "医院等级"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def append_cleaning_log(rows: list[dict[str, str]]) -> None:
    file_exists = CLEANING_LOG_PATH.exists()
    is_empty = (not file_exists) or CLEANING_LOG_PATH.stat().st_size == 0
    with CLEANING_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANING_LOG_COLUMNS)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)


def pick_column(columns: list[str], candidates: list[str], label: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(f"Could not find {label} column. Available columns: {columns}")


def clean_hospital_name(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def main() -> None:
    ensure_directories()

    analysis_df = pd.read_parquet(INPUT_ANALYSIS_PATH)
    hospital_df = pd.read_parquet(INPUT_HOSPITAL_PATH)

    hospital_columns = list(hospital_df.columns)
    hospital_col = pick_column(hospital_columns, HOSPITAL_COL_CANDIDATES, "hospital name")
    level_col = pick_column(hospital_columns, LEVEL_COL_CANDIDATES, "hospital level")

    hospital_lookup = hospital_df[[hospital_col, level_col]].copy()
    hospital_lookup = hospital_lookup.rename(
        columns={hospital_col: "hospital_from_lookup", level_col: "hospitallevel"}
    )
    hospital_lookup["hospital_key"] = clean_hospital_name(hospital_lookup["hospital_from_lookup"])
    hospital_lookup["hospitallevel"] = hospital_lookup["hospitallevel"].astype("string").str.strip()

    lookup_original_rows = int(len(hospital_lookup))
    lookup_non_empty_rows = int((hospital_lookup["hospital_key"] != "").sum())
    lookup_duplicates = int(hospital_lookup["hospital_key"].duplicated(keep=False).sum())

    hospital_lookup = (
        hospital_lookup.loc[hospital_lookup["hospital_key"] != ""]
        .drop_duplicates(subset=["hospital_key"], keep="first")
        .reset_index(drop=True)
    )

    analysis_df = analysis_df.copy()
    analysis_df["hospital_key"] = clean_hospital_name(analysis_df["hospital"])
    joined_df = analysis_df.merge(
        hospital_lookup[["hospital_key", "hospitallevel"]],
        on="hospital_key",
        how="left",
    )
    joined_df = joined_df.drop(columns=["hospital_key"])
    joined_df.to_parquet(OUTPUT_ANALYSIS_PATH, index=False)

    matched_row_mask = joined_df["hospitallevel"].notna() & (joined_df["hospitallevel"] != "")
    matched_rows = int(matched_row_mask.sum())
    total_rows = int(len(joined_df))

    distinct_hospital_df = (
        joined_df.loc[joined_df["hospital"].notna() & (joined_df["hospital"].astype("string").str.strip() != "")]
        [["hospital", "hospitallevel"]]
        .drop_duplicates(subset=["hospital"])
        .reset_index(drop=True)
    )
    distinct_hospitals = int(len(distinct_hospital_df))
    matched_distinct_hospitals = int(
        distinct_hospital_df["hospitallevel"].notna().sum()
    )

    distribution_df = (
        joined_df.assign(
            hospitallevel=joined_df["hospitallevel"].fillna("UNMATCHED")
        )["hospitallevel"]
        .value_counts(dropna=False)
        .rename_axis("hospitallevel")
        .reset_index(name="row_count")
    )
    distribution_df["row_share"] = (distribution_df["row_count"] / total_rows).round(6)

    distinct_distribution_df = (
        distinct_hospital_df.assign(
            hospitallevel=distinct_hospital_df["hospitallevel"].fillna("UNMATCHED")
        )["hospitallevel"]
        .value_counts(dropna=False)
        .rename_axis("hospitallevel")
        .reset_index(name="distinct_hospital_count")
    )
    distinct_distribution_df["distinct_hospital_share"] = (
        distinct_distribution_df["distinct_hospital_count"] / distinct_hospitals
    ).round(6)

    distribution_df = distribution_df.merge(
        distinct_distribution_df, on="hospitallevel", how="left"
    )
    distribution_df.to_csv(DISTRIBUTION_PATH, index=False, encoding="utf-8-sig")

    unmatched_df = (
        distinct_hospital_df.loc[distinct_hospital_df["hospitallevel"].isna(), ["hospital"]]
        .assign(note="exact_name_match_failed")
        .sort_values("hospital", kind="stable")
        .reset_index(drop=True)
    )
    unmatched_df.to_csv(UNMATCHED_PATH, index=False, encoding="utf-8-sig")

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only_core_physician_titles",
        "input_analysis_path": str(INPUT_ANALYSIS_PATH),
        "input_hospital_lookup_path": str(INPUT_HOSPITAL_PATH),
        "output_analysis_path": str(OUTPUT_ANALYSIS_PATH),
        "lookup_columns": {
            "hospital_column": hospital_col,
            "hospitallevel_column": level_col,
        },
        "lookup_quality": {
            "lookup_original_rows": lookup_original_rows,
            "lookup_non_empty_rows": lookup_non_empty_rows,
            "lookup_rows_with_duplicate_hospital_name": lookup_duplicates,
            "lookup_distinct_hospitals_after_dedup": int(len(hospital_lookup)),
        },
        "match_results": {
            "analysis_total_rows": total_rows,
            "matched_rows": matched_rows,
            "matched_row_rate": round(matched_rows / total_rows, 6) if total_rows else 0.0,
            "analysis_distinct_hospitals": distinct_hospitals,
            "matched_distinct_hospitals": matched_distinct_hospitals,
            "matched_distinct_hospital_rate": round(
                matched_distinct_hospitals / distinct_hospitals, 6
            ) if distinct_hospitals else 0.0,
        },
        "top_distribution_rows": distribution_df.head(20).to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "04b_join_hospital_level",
                "action": "join_hospitallevel_by_hospital_name_and_audit_coverage",
                "status": "completed",
                "target_path": str(OUTPUT_ANALYSIS_PATH),
                "details": (
                    f"matched_rows={matched_rows}; "
                    f"matched_row_rate={summary['match_results']['matched_row_rate']}; "
                    f"matched_distinct_hospitals={matched_distinct_hospitals}; "
                    f"matched_distinct_hospital_rate={summary['match_results']['matched_distinct_hospital_rate']}"
                ),
            }
        ]
    )

    print(f"Joined analysis table written to {OUTPUT_ANALYSIS_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Distribution written to {DISTRIBUTION_PATH}")
    print(f"Unmatched hospitals written to {UNMATCHED_PATH}")


if __name__ == "__main__":
    main()
