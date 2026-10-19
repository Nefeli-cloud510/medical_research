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
    / "focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_major5.parquet"
)
OUTPUT_PATH = ANALYSIS_DIR / "major5_text_cleaning_prep.parquet"
SUMMARY_PATH = AUDIT_DIR / "14_major5_text_cleaning_prep_summary.json"
MULTI_ANSWER_PAIRS_PATH = TABLES_DIR / "14_same_doctor_multi_answers_within_question.csv"
PAIR_COUNT_DISTRIBUTION_PATH = TABLES_DIR / "14_question_user_answer_count_distribution.csv"
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


def main() -> None:
    ensure_directories()
    df = pd.read_parquet(INPUT_PATH).copy()

    df["user_id"] = clean_string(df["user_id"])
    df["answer"] = clean_string(df["answer"])
    df["question_desc"] = clean_string(df["question_desc"])
    df["question_answer_join_ready"] = True

    # Use actual answer time and existing within-question ranks to stably order repeated answers
    # from the same doctor under the same question.
    df = df.sort_values(
        ["question_id", "user_id", "answer_create_time", "time_seq", "seq"],
        kind="stable",
    ).reset_index(drop=True)

    pair_group = df.groupby(["question_id", "user_id"], dropna=False)
    df["question_user_answer_count"] = pair_group["question_id"].transform("size").astype("int64")
    df["doctor_answer_order_within_question"] = pair_group.cumcount().add(1).astype("int64")
    df["is_same_doctor_multi_answer_in_question"] = df["question_user_answer_count"] > 1
    df["is_doctor_first_answer_in_question"] = df["doctor_answer_order_within_question"] == 1
    df["is_doctor_followup_answer_in_question"] = df["doctor_answer_order_within_question"] > 1

    df.to_parquet(OUTPUT_PATH, index=False)

    pair_summary_df = (
        df.groupby(["question_id", "user_id"], dropna=False)
        .agg(
            question_user_answer_count=("question_user_answer_count", "first"),
            first_answer_create_time=("answer_create_time", "min"),
            last_answer_create_time=("answer_create_time", "max"),
            first_time_seq=("time_seq", "min"),
            last_time_seq=("time_seq", "max"),
            sample_question_desc=("question_desc", "first"),
            sample_answer=("answer", "first"),
        )
        .reset_index()
    )

    multi_answer_pairs_df = (
        pair_summary_df.loc[pair_summary_df["question_user_answer_count"] > 1]
        .sort_values(
            ["question_user_answer_count", "question_id", "user_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    multi_answer_pairs_df.to_csv(MULTI_ANSWER_PAIRS_PATH, index=False, encoding="utf-8-sig")

    pair_count_distribution_df = (
        pair_summary_df["question_user_answer_count"]
        .value_counts(dropna=False)
        .rename_axis("question_user_answer_count")
        .reset_index(name="pair_count")
        .sort_values("question_user_answer_count", kind="stable")
        .reset_index(drop=True)
    )
    pair_count_distribution_df["pair_share"] = (
        pair_count_distribution_df["pair_count"] / len(pair_summary_df)
    ).round(6)
    pair_count_distribution_df.to_csv(PAIR_COUNT_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig")

    total_rows = int(len(df))
    distinct_question_user_pairs = int(len(pair_summary_df))
    multi_answer_pairs = int(len(multi_answer_pairs_df))
    rows_in_multi_answer_pairs = int(
        df.loc[df["is_same_doctor_multi_answer_in_question"]].shape[0]
    )
    followup_rows_same_doctor = int(df["is_doctor_followup_answer_in_question"].sum())
    max_answers_per_pair = int(pair_summary_df["question_user_answer_count"].max())

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "major5_main_regression_library_for_text_cleaning_and_actual_computation",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "join_status": {
            "question_answer_already_joined_in_input": True,
            "action_taken": "materialize_text_cleaning_prep_version_from_major5",
            "note": "major5 已包含 question_desc 与 answer 等问答拼接字段，本步不再回到原始表重复 join。",
        },
        "row_counts": {
            "total_rows": total_rows,
            "distinct_question_user_pairs": distinct_question_user_pairs,
            "rows_in_multi_answer_pairs": rows_in_multi_answer_pairs,
            "followup_rows_same_doctor": followup_rows_same_doctor,
        },
        "question_user_repeat_stats": {
            "multi_answer_pairs": multi_answer_pairs,
            "multi_answer_pair_rate": round(multi_answer_pairs / distinct_question_user_pairs, 6)
            if distinct_question_user_pairs
            else 0.0,
            "max_answers_per_pair": max_answers_per_pair,
        },
        "pair_count_distribution": pair_count_distribution_df.to_dict(orient="records"),
        "top_multi_answer_pairs": multi_answer_pairs_df.head(20).to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05_prepare_major5_text_base",
                "action": "materialize_major5_text_cleaning_prep_with_question_user_counts",
                "status": "completed",
                "target_path": str(OUTPUT_PATH),
                "details": (
                    f"input={INPUT_PATH.name}; total_rows={total_rows}; "
                    f"distinct_question_user_pairs={distinct_question_user_pairs}; "
                    f"multi_answer_pairs={multi_answer_pairs}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05_prepare_major5_text_base",
                "action": "write_question_user_repeat_audits",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"multi_pairs={MULTI_ANSWER_PAIRS_PATH.name}; "
                    f"distribution={PAIR_COUNT_DISTRIBUTION_PATH.name}; "
                    f"followup_rows_same_doctor={followup_rows_same_doctor}"
                ),
            },
        ]
    )

    print(f"Text cleaning prep written to {OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Multi-answer pair audit written to {MULTI_ANSWER_PAIRS_PATH}")
    print(f"Pair-count distribution written to {PAIR_COUNT_DISTRIBUTION_PATH}")


if __name__ == "__main__":
    main()
