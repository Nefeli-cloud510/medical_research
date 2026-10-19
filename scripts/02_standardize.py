from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
STAGING_DIR = ROOT_DIR / "data" / "staging"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

QUESTION_PATH = RAW_DIR / "question.parquet"
ANSWER_PATH = RAW_DIR / "answer.parquet"
OUTPUT_PATH = STAGING_DIR / "question_answer_joined.parquet"
SUMMARY_PATH = AUDIT_DIR / "04_question_answer_join_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

QUESTION_COLUMNS = [
    "question_id",
    "gender",
    "age",
    "address",
    "question_desc",
    "tags",
    "create_time",
    "bounty",
    "views",
    "ans_count",
]

ANSWER_COLUMNS = [
    "question_id",
    "user_id",
    "tech_title",
    "dept",
    "hospital",
    "answer",
    "create_time",
    "thumb",
    "help_status",
    "seq",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def append_cleaning_log(rows: list[dict[str, str]]) -> None:
    file_exists = CLEANING_LOG_PATH.exists()
    is_empty = (not file_exists) or CLEANING_LOG_PATH.stat().st_size == 0
    with CLEANING_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANING_LOG_COLUMNS)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)


def load_question() -> pd.DataFrame:
    question_df = pd.read_parquet(QUESTION_PATH, columns=QUESTION_COLUMNS)
    return question_df.rename(columns={"create_time": "question_create_time"})


def load_answer() -> pd.DataFrame:
    answer_df = pd.read_parquet(ANSWER_PATH, columns=ANSWER_COLUMNS)
    return answer_df.rename(columns={"create_time": "answer_create_time"})


def build_joined_table(question_df: pd.DataFrame, answer_df: pd.DataFrame) -> pd.DataFrame:
    joined_df = answer_df.merge(question_df, on="question_id", how="left", validate="many_to_one")
    ordered_columns = [
        "question_id",
        "seq",
        "user_id",
        "tech_title",
        "dept",
        "hospital",
        "answer",
        "thumb",
        "help_status",
        "answer_create_time",
        "gender",
        "age",
        "address",
        "question_desc",
        "tags",
        "bounty",
        "views",
        "ans_count",
        "question_create_time",
    ]
    return joined_df[ordered_columns].sort_values(["question_id", "seq", "answer_create_time"], kind="stable")


def build_summary(question_df: pd.DataFrame, answer_df: pd.DataFrame, joined_df: pd.DataFrame) -> dict:
    missing_question_rows = int(joined_df["question_desc"].isna().sum())
    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only",
        "inputs": {
            "question_path": str(QUESTION_PATH),
            "answer_path": str(ANSWER_PATH),
            "question_rows": int(len(question_df)),
            "answer_rows": int(len(answer_df)),
            "question_unique_ids": int(question_df["question_id"].nunique()),
            "answer_unique_question_ids": int(answer_df["question_id"].nunique()),
        },
        "output": {
            "joined_path": str(OUTPUT_PATH),
            "joined_rows": int(len(joined_df)),
            "joined_columns": list(joined_df.columns),
            "missing_question_side_rows": missing_question_rows,
            "question_match_rate_in_answers": round(
                float((joined_df["question_desc"].notna()).mean()),
                6,
            ),
        },
        "notes": [
            "This step only performs question-answer joining and column disambiguation.",
            "No record filtering, deduplication, text cleaning, or feature engineering is applied here.",
            "create_time is preserved as question_create_time and answer_create_time.",
        ],
    }
    return summary


def main() -> None:
    ensure_directories()
    question_df = load_question()
    answer_df = load_answer()
    joined_df = build_joined_table(question_df, answer_df)
    joined_df.to_parquet(OUTPUT_PATH, index=False)

    summary = build_summary(question_df, answer_df, joined_df)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "02_standardize",
                "action": "join_question_and_answer",
                "status": "completed",
                "target_path": str(OUTPUT_PATH),
                "details": (
                    f"question_rows={summary['inputs']['question_rows']}; "
                    f"answer_rows={summary['inputs']['answer_rows']}; "
                    f"joined_rows={summary['output']['joined_rows']}; "
                    f"missing_question_side_rows={summary['output']['missing_question_side_rows']}"
                ),
            }
        ]
    )

    print(f"Joined table written to {OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
