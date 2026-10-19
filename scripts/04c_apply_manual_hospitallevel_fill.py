from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

INPUT_ANALYSIS_PATH = ANALYSIS_DIR / "focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel.parquet"
MANUAL_FILL_PATH = TABLES_DIR / "08_unmatched_hospitals.csv"
OUTPUT_ANALYSIS_PATH = ANALYSIS_DIR / "focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_filled.parquet"
SUMMARY_PATH = AUDIT_DIR / "09_hospitallevel_manual_fill_summary.json"
DISTRIBUTION_PATH = TABLES_DIR / "09_hospitallevel_distribution_after_manual_fill.csv"
REMAINING_UNMATCHED_PATH = TABLES_DIR / "09_remaining_unmatched_hospitals.csv"
NONSTANDARD_PATH = TABLES_DIR / "09_nonstandard_hospitallevel_labels.csv"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

STANDARD_LEVEL_PATTERN = re.compile(r"^[一二三]级[甲乙丙]等$")


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


def is_nonempty(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype("string").str.strip() != "")


def is_standard_label(value: str) -> bool:
    return bool(STANDARD_LEVEL_PATTERN.fullmatch(value))


def main() -> None:
    ensure_directories()

    analysis_df = pd.read_parquet(INPUT_ANALYSIS_PATH)
    manual_df = pd.read_csv(MANUAL_FILL_PATH)

    if "hospital" not in manual_df.columns or "等级" not in manual_df.columns:
        raise KeyError("Manual fill file must contain 'hospital' and '等级' columns.")

    manual_df = manual_df.copy()
    manual_df["hospital_key"] = clean_string(manual_df["hospital"])
    manual_df["manual_hospitallevel"] = clean_string(manual_df["等级"])
    manual_df["manual_hospitallevel_source"] = (
        clean_string(manual_df["数据来源"]) if "数据来源" in manual_df.columns else ""
    )
    manual_df["manual_hospitallevel_note"] = (
        clean_string(manual_df["备注"]) if "备注" in manual_df.columns else ""
    )
    manual_df = manual_df.loc[manual_df["hospital_key"] != ""].copy()
    manual_df = manual_df.drop_duplicates(subset=["hospital_key"], keep="first").reset_index(drop=True)

    analysis_df = analysis_df.copy()
    analysis_df["hospital_key"] = clean_string(analysis_df["hospital"])
    analysis_df["hospitallevel"] = clean_string(analysis_df["hospitallevel"])
    analysis_df = analysis_df.merge(
        manual_df[
            [
                "hospital_key",
                "manual_hospitallevel",
                "manual_hospitallevel_source",
                "manual_hospitallevel_note",
            ]
        ],
        on="hospital_key",
        how="left",
    )

    original_matched_mask = analysis_df["hospitallevel"] != ""
    manual_nonempty_mask = analysis_df["manual_hospitallevel"] != ""
    fill_mask = (~original_matched_mask) & manual_nonempty_mask & (analysis_df["manual_hospitallevel"] != "UNMATCHED")

    analysis_df["hospitallevel_final"] = analysis_df["hospitallevel"]
    analysis_df.loc[fill_mask, "hospitallevel_final"] = analysis_df.loc[fill_mask, "manual_hospitallevel"]

    analysis_df["hospitallevel_fill_method"] = "lookup_exact_name"
    analysis_df.loc[fill_mask, "hospitallevel_fill_method"] = "manual_search_fill"
    analysis_df.loc[(analysis_df["hospitallevel_final"] == "") & manual_nonempty_mask, "hospitallevel_fill_method"] = "manual_review_still_unmatched"
    analysis_df.loc[(analysis_df["hospitallevel_final"] == "") & (~manual_nonempty_mask), "hospitallevel_fill_method"] = "unmatched"

    analysis_df["hospitallevel_source"] = ""
    analysis_df.loc[fill_mask | manual_nonempty_mask, "hospitallevel_source"] = analysis_df.loc[
        fill_mask | manual_nonempty_mask, "manual_hospitallevel_source"
    ]
    analysis_df["hospitallevel_note"] = ""
    analysis_df.loc[fill_mask | manual_nonempty_mask, "hospitallevel_note"] = analysis_df.loc[
        fill_mask | manual_nonempty_mask, "manual_hospitallevel_note"
    ]

    analysis_df["hospitallevel_final"] = analysis_df["hospitallevel_final"].replace("", pd.NA)
    analysis_df.to_parquet(OUTPUT_ANALYSIS_PATH, index=False)

    total_rows = int(len(analysis_df))
    matched_row_mask = analysis_df["hospitallevel_final"].notna()
    matched_rows = int(matched_row_mask.sum())
    unmatched_rows = total_rows - matched_rows
    manual_filled_rows = int(fill_mask.sum())

    distinct_hospital_df = (
        analysis_df.loc[analysis_df["hospital_key"] != "", ["hospital", "hospitallevel_final"]]
        .drop_duplicates(subset=["hospital"])
        .reset_index(drop=True)
    )
    distinct_hospitals = int(len(distinct_hospital_df))
    matched_distinct_hospitals = int(distinct_hospital_df["hospitallevel_final"].notna().sum())
    unmatched_distinct_hospitals = distinct_hospitals - matched_distinct_hospitals

    distribution_df = (
        analysis_df.assign(
            hospitallevel_final=analysis_df["hospitallevel_final"].fillna("UNMATCHED")
        )["hospitallevel_final"]
        .value_counts(dropna=False)
        .rename_axis("hospitallevel")
        .reset_index(name="row_count")
    )
    distribution_df["row_share"] = (distribution_df["row_count"] / total_rows).round(6)

    distinct_distribution_df = (
        distinct_hospital_df.assign(
            hospitallevel_final=distinct_hospital_df["hospitallevel_final"].fillna("UNMATCHED")
        )["hospitallevel_final"]
        .value_counts(dropna=False)
        .rename_axis("hospitallevel")
        .reset_index(name="distinct_hospital_count")
    )
    distinct_distribution_df["distinct_hospital_share"] = (
        distinct_distribution_df["distinct_hospital_count"] / distinct_hospitals
    ).round(6)

    distribution_df = distribution_df.merge(distinct_distribution_df, on="hospitallevel", how="left")
    distribution_df.to_csv(DISTRIBUTION_PATH, index=False, encoding="utf-8-sig")

    remaining_unmatched_df = (
        distinct_hospital_df.loc[distinct_hospital_df["hospitallevel_final"].isna(), ["hospital"]]
        .assign(note="still_unmatched_after_manual_fill")
        .sort_values("hospital", kind="stable")
        .reset_index(drop=True)
    )
    remaining_unmatched_df.to_csv(REMAINING_UNMATCHED_PATH, index=False, encoding="utf-8-sig")

    level_status_df = distribution_df.copy()
    level_status_df["is_standard_label"] = level_status_df["hospitallevel"].map(
        lambda x: is_standard_label(str(x))
    )
    level_status_df["label_type"] = level_status_df["is_standard_label"].map(
        {True: "standard_graded", False: "nonstandard_or_unmatched"}
    )
    nonstandard_df = level_status_df.loc[~level_status_df["is_standard_label"]].copy()
    nonstandard_df.to_csv(NONSTANDARD_PATH, index=False, encoding="utf-8-sig")

    nonstandard_rows = int(nonstandard_df["row_count"].sum())
    nonstandard_distinct_hospitals = int(nonstandard_df["distinct_hospital_count"].sum())

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only_core_physician_titles",
        "input_analysis_path": str(INPUT_ANALYSIS_PATH),
        "manual_fill_path": str(MANUAL_FILL_PATH),
        "output_analysis_path": str(OUTPUT_ANALYSIS_PATH),
        "manual_fill_quality": {
            "manual_rows_after_dedup": int(len(manual_df)),
            "manual_rows_with_nonempty_level": int((manual_df["manual_hospitallevel"] != "").sum()),
            "manual_rows_marked_unmatched": int((manual_df["manual_hospitallevel"] == "UNMATCHED").sum()),
        },
        "match_results_after_fill": {
            "analysis_total_rows": total_rows,
            "matched_rows": matched_rows,
            "matched_row_rate": round(matched_rows / total_rows, 6) if total_rows else 0.0,
            "unmatched_rows": unmatched_rows,
            "unmatched_row_rate": round(unmatched_rows / total_rows, 6) if total_rows else 0.0,
            "manual_filled_rows": manual_filled_rows,
            "analysis_distinct_hospitals": distinct_hospitals,
            "matched_distinct_hospitals": matched_distinct_hospitals,
            "matched_distinct_hospital_rate": round(
                matched_distinct_hospitals / distinct_hospitals, 6
            ) if distinct_hospitals else 0.0,
            "unmatched_distinct_hospitals": unmatched_distinct_hospitals,
            "unmatched_distinct_hospital_rate": round(
                unmatched_distinct_hospitals / distinct_hospitals, 6
            ) if distinct_hospitals else 0.0,
        },
        "nonstandard_label_results": {
            "nonstandard_rows": nonstandard_rows,
            "nonstandard_row_rate": round(nonstandard_rows / total_rows, 6) if total_rows else 0.0,
            "nonstandard_distinct_hospitals": nonstandard_distinct_hospitals,
            "nonstandard_distinct_hospital_rate": round(
                nonstandard_distinct_hospitals / distinct_hospitals, 6
            ) if distinct_hospitals else 0.0,
        },
        "top_distribution_rows": distribution_df.head(30).to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "04c_apply_manual_hospitallevel_fill",
                "action": "apply_manual_hospitallevel_fill_and_reaudit_distribution",
                "status": "completed",
                "target_path": str(OUTPUT_ANALYSIS_PATH),
                "details": (
                    f"matched_rows={matched_rows}; "
                    f"unmatched_rows={unmatched_rows}; "
                    f"manual_filled_rows={manual_filled_rows}; "
                    f"nonstandard_rows={nonstandard_rows}"
                ),
            }
        ]
    )

    print(f"Filled analysis table written to {OUTPUT_ANALYSIS_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Distribution written to {DISTRIBUTION_PATH}")
    print(f"Remaining unmatched hospitals written to {REMAINING_UNMATCHED_PATH}")
    print(f"Nonstandard labels written to {NONSTANDARD_PATH}")


if __name__ == "__main__":
    main()
