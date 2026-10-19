from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
STAGING_DIR = ROOT_DIR / "data" / "staging"
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

INPUT_PATH = STAGING_DIR / "question_answer_joined.parquet"
OUTPUT_PATH = ANALYSIS_DIR / "focal_answer_panel_base.parquet"
SUMMARY_PATH = AUDIT_DIR / "05_time_and_prior_answer_checks.json"
INVALID_TIME_SAMPLE_PATH = AUDIT_DIR / "05_invalid_time_samples.csv"
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


def load_joined_table() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    return df.sort_values(["question_id", "seq", "answer_create_time"], kind="stable").reset_index(drop=True)


def build_prior_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_sorted_index = df.sort_values(["question_id", "answer_create_time", "seq"], kind="stable").index
    time_seq_series = pd.Series(
        df.loc[time_sorted_index].groupby("question_id").cumcount().add(1).to_numpy(),
        index=time_sorted_index,
    ).reindex(df.index)
    df["time_seq"] = time_seq_series.astype("int64")
    df["observed_seq_by_time"] = df["time_seq"]
    df["prior_answer_count"] = df["seq"] - 1
    df["prior_answer_count_by_time"] = df["time_seq"] - 1
    df["answer_delay_hours"] = (
        (df["answer_create_time"] - df["question_create_time"]).dt.total_seconds() / 3600
    )
    df["is_answer_time_valid"] = df["answer_delay_hours"].ge(0)
    df["is_seq_time_aligned"] = df["seq"].eq(df["time_seq"])
    df["time_rank_within_question"] = df["time_seq"]
    df["has_prior_answers"] = df["prior_answer_count"].ge(1)
    return df


def export_invalid_time_samples(df: pd.DataFrame) -> int:
    invalid_df = (
        df.loc[
            ~df["is_answer_time_valid"],
            [
                "question_id",
                "seq",
                "time_seq",
                "user_id",
                "tech_title",
                "dept",
                "hospital",
                "question_create_time",
                "answer_create_time",
                "answer_delay_hours",
                "question_desc",
                "answer",
            ],
        ]
        .sort_values(["question_id", "time_seq", "answer_create_time"], kind="stable")
        .reset_index(drop=True)
    )
    invalid_df.to_csv(INVALID_TIME_SAMPLE_PATH, index=False, encoding="utf-8-sig")
    return int(len(invalid_df))


def build_summary(df: pd.DataFrame) -> dict:
    total_rows = int(len(df))
    invalid_time_mask = ~df["is_answer_time_valid"]
    seq_mismatch_mask = ~df["is_seq_time_aligned"]
    missing_question_time_mask = df["question_create_time"].isna()
    missing_answer_time_mask = df["answer_create_time"].isna()

    invalid_time_rows = int(invalid_time_mask.sum())
    seq_mismatch_rows = int(seq_mismatch_mask.sum())
    multi_answer_rows = int(df["has_prior_answers"].sum())

    seq_mismatch_examples = (
        df.loc[seq_mismatch_mask, ["question_id", "seq", "observed_seq_by_time", "answer_create_time"]]
        .head(20)
        .assign(answer_create_time=lambda x: x["answer_create_time"].astype(str))
        .to_dict(orient="records")
    )
    invalid_time_examples = (
        df.loc[invalid_time_mask, ["question_id", "seq", "question_create_time", "answer_create_time", "answer_delay_hours"]]
        .head(20)
        .assign(
            question_create_time=lambda x: x["question_create_time"].astype(str),
            answer_create_time=lambda x: x["answer_create_time"].astype(str),
        )
        .to_dict(orient="records")
    )

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "row_count": total_rows,
        "checks": {
            "missing_question_time_rows": int(missing_question_time_mask.sum()),
            "missing_answer_time_rows": int(missing_answer_time_mask.sum()),
            "invalid_time_rows": invalid_time_rows,
            "invalid_time_rate": round(invalid_time_rows / total_rows, 6) if total_rows else 0.0,
            "seq_time_mismatch_rows": seq_mismatch_rows,
            "seq_time_mismatch_rate": round(seq_mismatch_rows / total_rows, 6) if total_rows else 0.0,
            "rows_with_prior_answers": multi_answer_rows,
            "rows_with_prior_answers_rate": round(multi_answer_rows / total_rows, 6) if total_rows else 0.0,
            "median_answer_delay_hours": round(float(df["answer_delay_hours"].median()), 6),
            "p95_answer_delay_hours": round(float(df["answer_delay_hours"].quantile(0.95)), 6),
        },
        "example_rows": {
            "invalid_time_examples": invalid_time_examples,
            "seq_time_mismatch_examples": seq_mismatch_examples,
        },
        "notes": [
            "This step checks whether answer_create_time is not earlier than question_create_time.",
            "This step checks whether seq matches the within-question order implied by answer_create_time.",
            "This step constructs only prior-answer indexing fields and does not perform text cleaning yet.",
        ],
    }
    return summary


def main() -> None:
    ensure_directories()
    df = load_joined_table()
    df = build_prior_fields(df)
    df.to_parquet(OUTPUT_PATH, index=False)
    invalid_time_exported_rows = export_invalid_time_samples(df)

    summary = build_summary(df)
    summary["outputs"] = {
        "analysis_base_path": str(OUTPUT_PATH),
        "invalid_time_sample_path": str(INVALID_TIME_SAMPLE_PATH),
        "invalid_time_exported_rows": invalid_time_exported_rows,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "03_clean_answers",
                "action": "time_order_checks_and_prior_answer_indexing",
                "status": "completed",
                "target_path": str(OUTPUT_PATH),
                "details": (
                    f"row_count={summary['row_count']}; "
                    f"invalid_time_rows={summary['checks']['invalid_time_rows']}; "
                    f"invalid_time_exported_rows={invalid_time_exported_rows}; "
                    f"seq_time_mismatch_rows={summary['checks']['seq_time_mismatch_rows']}; "
                    f"rows_with_prior_answers={summary['checks']['rows_with_prior_answers']}"
                ),
            }
        ]
    )

    print(f"Analysis base table written to {OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
