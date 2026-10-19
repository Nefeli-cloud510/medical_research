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

INPUT_PATH = ANALYSIS_DIR / "major5_text_cleaning_prep_seq_clean.parquet"
SUMMARY_PATH = AUDIT_DIR / "16_major5_strict_prior_audit_summary.json"
STRICT_PRIOR_DISTRIBUTION_PATH = TABLES_DIR / "16_strict_prior_count_distribution_major5.csv"
SAME_TIMESTAMP_GROUPS_PATH = TABLES_DIR / "16_same_timestamp_groups_major5.csv"
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


def main() -> None:
    ensure_directories()
    df = pd.read_parquet(INPUT_PATH, columns=["question_id", "answer_create_time", "time_seq", "seq"]).copy()

    total_rows = int(len(df))
    total_questions = int(df["question_id"].nunique())

    timestamp_group_df = (
        df.groupby(["question_id", "answer_create_time"], dropna=False)
        .size()
        .reset_index(name="same_timestamp_group_size")
    )
    timestamp_group_df = timestamp_group_df.sort_values(
        ["question_id", "answer_create_time"],
        kind="stable",
    ).reset_index(drop=True)
    timestamp_group_df["strict_prior_count_current"] = (
        timestamp_group_df.groupby("question_id")["same_timestamp_group_size"]
        .transform(lambda series: series.cumsum().shift(fill_value=0))
        .astype("int64")
    )

    df = df.merge(
        timestamp_group_df,
        on=["question_id", "answer_create_time"],
        how="left",
    )

    strict_prior_distribution_df = (
        df["strict_prior_count_current"]
        .value_counts(dropna=False)
        .rename_axis("strict_prior_count_current")
        .reset_index(name="answer_count")
        .sort_values("strict_prior_count_current", kind="stable")
        .reset_index(drop=True)
    )
    strict_prior_distribution_df["answer_share"] = (
        strict_prior_distribution_df["answer_count"] / total_rows
    ).round(6)
    strict_prior_distribution_df.to_csv(
        STRICT_PRIOR_DISTRIBUTION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    same_timestamp_groups_df = (
        timestamp_group_df.loc[timestamp_group_df["same_timestamp_group_size"] > 1]
        .sort_values(
            ["same_timestamp_group_size", "question_id", "answer_create_time"],
            ascending=[False, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    same_timestamp_groups_df.to_csv(SAME_TIMESTAMP_GROUPS_PATH, index=False, encoding="utf-8-sig")

    same_timestamp_answer_rows = int(
        df.loc[df["same_timestamp_group_size"] > 1].shape[0]
    )
    same_timestamp_question_count = int(
        same_timestamp_groups_df["question_id"].nunique()
    )
    same_timestamp_group_count = int(len(same_timestamp_groups_df))
    max_same_timestamp_group_size = int(
        same_timestamp_groups_df["same_timestamp_group_size"].max()
    ) if same_timestamp_group_count else 1

    strict_prior_at_least_1 = int((df["strict_prior_count_current"] >= 1).sum())
    strict_prior_at_least_2 = int((df["strict_prior_count_current"] >= 2).sum())

    top_same_timestamp_groups = same_timestamp_groups_df.head(20).copy()
    if not top_same_timestamp_groups.empty:
        top_same_timestamp_groups["answer_create_time"] = top_same_timestamp_groups[
            "answer_create_time"
        ].astype(str)

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "major5_text_cleaning_prep_seq_clean_strict_prior_audit",
        "input_path": str(INPUT_PATH),
        "totals": {
            "total_rows": total_rows,
            "total_questions": total_questions,
        },
        "same_timestamp_stats_within_question": {
            "same_timestamp_answer_rows": same_timestamp_answer_rows,
            "same_timestamp_answer_row_rate": round(same_timestamp_answer_rows / total_rows, 6)
            if total_rows
            else 0.0,
            "same_timestamp_question_count": same_timestamp_question_count,
            "same_timestamp_question_rate": round(same_timestamp_question_count / total_questions, 6)
            if total_questions
            else 0.0,
            "same_timestamp_group_count": same_timestamp_group_count,
            "max_same_timestamp_group_size": max_same_timestamp_group_size,
        },
        "strict_prior_stats_current_sample": {
            "strict_prior_at_least_1_answer_count": strict_prior_at_least_1,
            "strict_prior_at_least_1_answer_rate": round(strict_prior_at_least_1 / total_rows, 6)
            if total_rows
            else 0.0,
            "strict_prior_at_least_2_answer_count": strict_prior_at_least_2,
            "strict_prior_at_least_2_answer_rate": round(strict_prior_at_least_2 / total_rows, 6)
            if total_rows
            else 0.0,
        },
        "top_same_timestamp_groups": top_same_timestamp_groups.to_dict(orient="records"),
        "strict_prior_distribution_preview": strict_prior_distribution_df.head(20).to_dict(orient="records"),
        "note": "严格前序人数按当前 major5 样本重新计算；同一 question_id 内时间戳完全相同的回答互不计入彼此的严格前序。",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05c_audit_major5_strict_prior",
                "action": "audit_same_timestamp_and_strict_prior_counts",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"total_rows={total_rows}; same_timestamp_answer_rows={same_timestamp_answer_rows}; "
                    f"same_timestamp_questions={same_timestamp_question_count}; "
                    f"strict_prior_ge1={strict_prior_at_least_1}; strict_prior_ge2={strict_prior_at_least_2}"
                ),
            }
        ]
    )

    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Strict prior distribution written to {STRICT_PRIOR_DISTRIBUTION_PATH}")
    print(f"Same timestamp groups written to {SAME_TIMESTAMP_GROUPS_PATH}")


if __name__ == "__main__":
    main()
