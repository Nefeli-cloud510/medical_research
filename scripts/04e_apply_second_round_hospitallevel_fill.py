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

INPUT_ANALYSIS_PATH = (
    ANALYSIS_DIR
    / "focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_filled.parquet"
)
MANUAL_FILL_PATH = TABLES_DIR / "10_nonstandard_hospital_levels_for_manual_search.xlsx"
OUTPUT_ANALYSIS_PATH = (
    ANALYSIS_DIR
    / "focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_filled_round2.parquet"
)
SUMMARY_PATH = AUDIT_DIR / "11_hospitallevel_second_round_fill_summary.json"
DISTRIBUTION_PATH = TABLES_DIR / "11_hospitallevel_distribution_after_round2_fill.csv"
REMAINING_UNMATCHED_PATH = TABLES_DIR / "11_remaining_unmatched_hospitals.csv"
NONSTANDARD_PATH = TABLES_DIR / "11_nonstandard_hospitallevel_labels.csv"
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

LEVEL_COLUMN_CANDIDATES = ["hospitallevel_final", "等级"]
SOURCE_COLUMN_CANDIDATES = ["hospitallevel_source", "数据来源"]
NOTE_COLUMN_CANDIDATES = ["hospitallevel_note", "备注"]
METHOD_COLUMN_CANDIDATES = ["hospitallevel_fill_method"]


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


def load_manual_fill(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path).copy()

    if suffix in {".xlsx", ".xls"}:
        raw_df = pd.read_excel(path, header=None).copy()
        if raw_df.empty:
            return pd.DataFrame(columns=["hospital", "hospitallevel_final", "hospitallevel_note"])

        raw_df = raw_df.rename(columns={0: "hospital", 1: "hospitallevel_final", 2: "hospitallevel_note"})
        keep_columns = ["hospital", "hospitallevel_final", "hospitallevel_note"]
        for column in keep_columns:
            if column not in raw_df.columns:
                raw_df[column] = ""
        return raw_df[keep_columns].copy()

    raise ValueError(f"Unsupported manual fill file format: {path}")


def pick_column(columns: list[str], candidates: list[str]) -> str | None:
    for column in candidates:
        if column in columns:
            return column
    return None


def is_standard_label(value: str) -> bool:
    return bool(STANDARD_LEVEL_PATTERN.fullmatch(value))


def build_distribution(
    analysis_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, int]:
    total_rows = int(len(analysis_df))
    matched_row_mask = analysis_df["hospitallevel_final"].notna()
    matched_rows = int(matched_row_mask.sum())
    unmatched_rows = total_rows - matched_rows

    distinct_hospital_df = (
        analysis_df.loc[analysis_df["hospital_key"] != "", ["hospital", "hospitallevel_final"]]
        .drop_duplicates(subset=["hospital"])
        .reset_index(drop=True)
    )
    distinct_hospitals = int(len(distinct_hospital_df))

    distribution_df = (
        analysis_df.assign(hospitallevel_final=analysis_df["hospitallevel_final"].fillna("UNMATCHED"))[
            "hospitallevel_final"
        ]
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
    return distribution_df, distinct_hospital_df, total_rows, matched_rows, unmatched_rows


def main() -> None:
    ensure_directories()

    analysis_df = pd.read_parquet(INPUT_ANALYSIS_PATH).copy()
    manual_df = load_manual_fill(MANUAL_FILL_PATH)

    if "hospital" not in manual_df.columns:
        raise KeyError("Manual fill file must contain a 'hospital' column.")

    level_col = pick_column(list(manual_df.columns), LEVEL_COLUMN_CANDIDATES)
    if level_col is None:
        raise KeyError(
            "Manual fill file must contain one of the level columns: "
            + ", ".join(LEVEL_COLUMN_CANDIDATES)
        )

    source_col = pick_column(list(manual_df.columns), SOURCE_COLUMN_CANDIDATES)
    note_col = pick_column(list(manual_df.columns), NOTE_COLUMN_CANDIDATES)
    method_col = pick_column(list(manual_df.columns), METHOD_COLUMN_CANDIDATES)

    manual_df["hospital_key"] = clean_string(manual_df["hospital"])
    manual_df["manual_hospitallevel"] = clean_string(manual_df[level_col])
    manual_df["manual_hospitallevel_source"] = (
        clean_string(manual_df[source_col]) if source_col else ""
    )
    manual_df["manual_hospitallevel_note"] = clean_string(manual_df[note_col]) if note_col else ""
    manual_df["manual_hospitallevel_fill_method"] = (
        clean_string(manual_df[method_col]) if method_col else ""
    )
    manual_df = manual_df.loc[manual_df["hospital_key"] != ""].copy()
    manual_df = manual_df.drop_duplicates(subset=["hospital_key"], keep="last").reset_index(drop=True)

    manual_df["has_round2_override"] = (
        (manual_df["manual_hospitallevel_fill_method"] == "manual_search_fill")
        | (manual_df["manual_hospitallevel_source"] != "")
        | (manual_df["manual_hospitallevel_note"] != "")
    )

    analysis_df["hospital_key"] = clean_string(analysis_df["hospital"])
    existing_manual_columns = [
        "manual_hospitallevel",
        "manual_hospitallevel_source",
        "manual_hospitallevel_note",
        "has_round2_override",
    ]
    analysis_df = analysis_df.drop(
        columns=[column for column in existing_manual_columns if column in analysis_df.columns]
    )
    for column in ["hospitallevel_final", "hospitallevel_source", "hospitallevel_note", "hospitallevel_fill_method"]:
        if column not in analysis_df.columns:
            analysis_df[column] = ""

    analysis_df["hospitallevel_final"] = clean_string(analysis_df["hospitallevel_final"])
    analysis_df["hospitallevel_source"] = clean_string(analysis_df["hospitallevel_source"])
    analysis_df["hospitallevel_note"] = clean_string(analysis_df["hospitallevel_note"])
    analysis_df["hospitallevel_fill_method"] = clean_string(analysis_df["hospitallevel_fill_method"])

    analysis_df = analysis_df.merge(
        manual_df[
            [
                "hospital_key",
                "manual_hospitallevel",
                "manual_hospitallevel_source",
                "manual_hospitallevel_note",
                "has_round2_override",
            ]
        ],
        on="hospital_key",
        how="left",
    )

    round2_mask = analysis_df["has_round2_override"].fillna(False)
    level_nonempty_mask = analysis_df["manual_hospitallevel"] != ""
    level_update_mask = round2_mask & level_nonempty_mask
    level_changed_mask = level_update_mask & (
        analysis_df["hospitallevel_final"] != analysis_df["manual_hospitallevel"]
    )
    source_update_mask = round2_mask & (analysis_df["manual_hospitallevel_source"] != "")
    note_update_mask = round2_mask & (analysis_df["manual_hospitallevel_note"] != "")

    analysis_df.loc[level_update_mask, "hospitallevel_final"] = analysis_df.loc[
        level_update_mask, "manual_hospitallevel"
    ]
    analysis_df.loc[source_update_mask, "hospitallevel_source"] = analysis_df.loc[
        source_update_mask, "manual_hospitallevel_source"
    ]
    analysis_df.loc[note_update_mask, "hospitallevel_note"] = analysis_df.loc[
        note_update_mask, "manual_hospitallevel_note"
    ]
    analysis_df.loc[round2_mask, "hospitallevel_fill_method"] = "manual_search_fill_round2"

    analysis_df["hospitallevel_final"] = analysis_df["hospitallevel_final"].replace("", pd.NA)
    analysis_df.to_parquet(OUTPUT_ANALYSIS_PATH, index=False)

    distribution_df, distinct_hospital_df, total_rows, matched_rows, unmatched_rows = build_distribution(
        analysis_df
    )
    distinct_hospitals = int(len(distinct_hospital_df))
    matched_distinct_hospitals = int(distinct_hospital_df["hospitallevel_final"].notna().sum())
    unmatched_distinct_hospitals = distinct_hospitals - matched_distinct_hospitals
    distribution_df.to_csv(DISTRIBUTION_PATH, index=False, encoding="utf-8-sig")

    remaining_unmatched_df = (
        distinct_hospital_df.loc[distinct_hospital_df["hospitallevel_final"].isna(), ["hospital"]]
        .assign(note="still_unmatched_after_round2_fill")
        .sort_values("hospital", kind="stable")
        .reset_index(drop=True)
    )
    remaining_unmatched_df.to_csv(REMAINING_UNMATCHED_PATH, index=False, encoding="utf-8-sig")

    level_status_df = distribution_df.copy()
    level_status_df["is_standard_label"] = level_status_df["hospitallevel"].map(
        lambda value: is_standard_label(str(value))
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
        "scope": "answer_question_only_core_physician_titles_second_round_nonstandard_fill",
        "input_analysis_path": str(INPUT_ANALYSIS_PATH),
        "manual_fill_path": str(MANUAL_FILL_PATH),
        "output_analysis_path": str(OUTPUT_ANALYSIS_PATH),
        "manual_fill_quality": {
            "manual_rows_after_dedup": int(len(manual_df)),
            "manual_rows_with_nonempty_level": int((manual_df["manual_hospitallevel"] != "").sum()),
            "manual_rows_with_round2_override": int(manual_df["has_round2_override"].sum()),
        },
        "round2_update_results": {
            "round2_touched_rows": int(round2_mask.sum()),
            "round2_touched_distinct_hospitals": int(
                manual_df.loc[manual_df["has_round2_override"], "hospital_key"].nunique()
            ),
            "level_updated_rows": int(level_update_mask.sum()),
            "level_changed_rows": int(level_changed_mask.sum()),
            "source_updated_rows": int(source_update_mask.sum()),
            "note_updated_rows": int(note_update_mask.sum()),
        },
        "match_results_after_fill": {
            "analysis_total_rows": total_rows,
            "matched_rows": matched_rows,
            "matched_row_rate": round(matched_rows / total_rows, 6) if total_rows else 0.0,
            "unmatched_rows": unmatched_rows,
            "unmatched_row_rate": round(unmatched_rows / total_rows, 6) if total_rows else 0.0,
            "analysis_distinct_hospitals": distinct_hospitals,
            "matched_distinct_hospitals": matched_distinct_hospitals,
            "matched_distinct_hospital_rate": round(
                matched_distinct_hospitals / distinct_hospitals, 6
            )
            if distinct_hospitals
            else 0.0,
            "unmatched_distinct_hospitals": unmatched_distinct_hospitals,
            "unmatched_distinct_hospital_rate": round(
                unmatched_distinct_hospitals / distinct_hospitals, 6
            )
            if distinct_hospitals
            else 0.0,
        },
        "nonstandard_label_results": {
            "nonstandard_rows": nonstandard_rows,
            "nonstandard_row_rate": round(nonstandard_rows / total_rows, 6) if total_rows else 0.0,
            "nonstandard_distinct_hospitals": nonstandard_distinct_hospitals,
            "nonstandard_distinct_hospital_rate": round(
                nonstandard_distinct_hospitals / distinct_hospitals, 6
            )
            if distinct_hospitals
            else 0.0,
        },
        "top_distribution_rows": distribution_df.head(20).to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_ts = summary["run_ts_utc"]
    append_cleaning_log(
        [
            {
                "run_ts_utc": run_ts,
                "step": "04e_apply_second_round_hospitallevel_fill",
                "action": "apply_second_round_manual_hospitallevel_fill",
                "status": "success",
                "target_path": str(OUTPUT_ANALYSIS_PATH),
                "details": (
                    f"input={INPUT_ANALYSIS_PATH.name}; manual={MANUAL_FILL_PATH.name}; "
                    f"round2_touched_rows={int(round2_mask.sum())}; "
                    f"level_changed_rows={int(level_changed_mask.sum())}"
                ),
            },
            {
                "run_ts_utc": run_ts,
                "step": "04e_apply_second_round_hospitallevel_fill",
                "action": "write_distribution_outputs",
                "status": "success",
                "target_path": str(DISTRIBUTION_PATH),
                "details": (
                    f"unmatched_rows={unmatched_rows}; "
                    f"nonstandard_rows={nonstandard_rows}; "
                    f"remaining_unmatched_hospitals={unmatched_distinct_hospitals}"
                ),
            },
            {
                "run_ts_utc": run_ts,
                "step": "04e_apply_second_round_hospitallevel_fill",
                "action": "write_summary_outputs",
                "status": "success",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"distribution={DISTRIBUTION_PATH.name}; "
                    f"remaining_unmatched={REMAINING_UNMATCHED_PATH.name}; "
                    f"nonstandard={NONSTANDARD_PATH.name}"
                ),
            },
        ]
    )


if __name__ == "__main__":
    main()
