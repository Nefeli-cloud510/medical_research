from __future__ import annotations

import csv
import hashlib
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

INPUT_PATH = (
    ANALYSIS_DIR
    / "focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_filled_round2.parquet"
)
OUTPUT_PATH = ANALYSIS_DIR / "base_answer_library_core_physician_round2.parquet"
SUMMARY_PATH = AUDIT_DIR / "13_base_answer_library_summary.json"
DUPLICATE_PATH = TABLES_DIR / "13_same_record_duplicate_captures.csv"
TEMPLATE_PATH = TABLES_DIR / "13_template_text_reuse_summary.csv"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

WHITESPACE_PATTERN = re.compile(r"\s+")


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


def normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text.strip())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_proxy_key(question_id: int, user_id: str, answer_create_time: str, answer_hash_raw: str) -> str:
    payload = f"{question_id}|{user_id}|{answer_create_time}|{answer_hash_raw}"
    return sha256_text(payload)


def main() -> None:
    ensure_directories()
    df = pd.read_parquet(INPUT_PATH).copy()

    df["user_id"] = clean_string(df["user_id"])
    df["answer"] = clean_string(df["answer"])
    df["answer_create_time_iso"] = pd.to_datetime(df["answer_create_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    df["answer_text_hash_raw"] = df["answer"].map(sha256_text)
    df["answer_text_normalized"] = df["answer"].map(normalize_text)
    df["answer_text_hash_norm"] = df["answer_text_normalized"].map(sha256_text)
    df["answer_event_proxy_key"] = [
        build_proxy_key(question_id, user_id, answer_create_time_iso, answer_hash_raw)
        for question_id, user_id, answer_create_time_iso, answer_hash_raw in zip(
            df["question_id"],
            df["user_id"],
            df["answer_create_time_iso"],
            df["answer_text_hash_raw"],
            strict=False,
        )
    ]
    df["same_record_duplicate_group_size"] = (
        df.groupby("answer_event_proxy_key")["answer_event_proxy_key"].transform("size").astype("int64")
    )
    df["is_same_record_duplicate_capture"] = df.duplicated(
        subset=["answer_event_proxy_key"],
        keep="first",
    )
    df["is_first_answer_by_time"] = df["time_seq"] == 1
    df["is_followup_answer_by_time"] = df["time_seq"] > 1

    original_rows = int(len(df))
    duplicate_rows = int(df["is_same_record_duplicate_capture"].sum())
    duplicate_groups = int((df["same_record_duplicate_group_size"] > 1).sum() - duplicate_rows)

    duplicate_capture_df = (
        df.loc[df["same_record_duplicate_group_size"] > 1, [
            "question_id",
            "user_id",
            "seq",
            "time_seq",
            "answer_create_time",
            "answer_event_proxy_key",
            "same_record_duplicate_group_size",
            "is_same_record_duplicate_capture",
            "answer",
        ]]
        .sort_values(
            ["answer_event_proxy_key", "is_same_record_duplicate_capture", "question_id", "user_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    duplicate_capture_df.to_csv(DUPLICATE_PATH, index=False, encoding="utf-8-sig")

    base_df = df.loc[~df["is_same_record_duplicate_capture"]].copy()
    deduped_rows = int(len(base_df))

    template_stats = (
        base_df.groupby("answer_text_hash_norm", dropna=False)
        .agg(
            template_occurrence_count=("answer_text_hash_norm", "size"),
            template_distinct_question_count=("question_id", "nunique"),
            template_distinct_user_count=("user_id", "nunique"),
            template_example_text=("answer", "first"),
        )
        .reset_index()
    )
    template_stats["is_template_text_reused"] = (
        (template_stats["template_occurrence_count"] > 1)
        & (
            (template_stats["template_distinct_question_count"] > 1)
            | (template_stats["template_distinct_user_count"] > 1)
        )
    )
    template_reuse_df = template_stats.loc[template_stats["is_template_text_reused"]].copy()
    template_reuse_df = template_reuse_df.sort_values(
        ["template_occurrence_count", "template_distinct_question_count", "template_distinct_user_count"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    template_reuse_df.to_csv(TEMPLATE_PATH, index=False, encoding="utf-8-sig")

    base_df = base_df.merge(
        template_stats[
            [
                "answer_text_hash_norm",
                "template_occurrence_count",
                "template_distinct_question_count",
                "template_distinct_user_count",
                "is_template_text_reused",
            ]
        ],
        on="answer_text_hash_norm",
        how="left",
    )
    base_df["template_occurrence_count"] = base_df["template_occurrence_count"].astype("int64")
    base_df["template_distinct_question_count"] = base_df["template_distinct_question_count"].astype("int64")
    base_df["template_distinct_user_count"] = base_df["template_distinct_user_count"].astype("int64")
    base_df["is_template_text_reused"] = base_df["is_template_text_reused"].fillna(False)

    cols_to_drop = ["answer_text_normalized"]
    base_df = base_df.drop(columns=[col for col in cols_to_drop if col in base_df.columns])
    base_df.to_parquet(OUTPUT_PATH, index=False)

    first_answer_rows = int((base_df["time_seq"] == 1).sum())
    followup_rows = int((base_df["time_seq"] > 1).sum())

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only_core_physician_titles_all_traceable_answers_round2",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "identity_strategy": {
            "real_answer_id_present_in_input": False,
            "proxy_key_name": "answer_event_proxy_key",
            "proxy_key_components": [
                "question_id",
                "user_id",
                "answer_create_time_iso",
                "answer_text_hash_raw",
            ],
            "note": "代理事件键用于复现与定位，不代表平台真实回答ID。",
        },
        "row_counts": {
            "original_rows": original_rows,
            "same_record_duplicate_rows_removed": duplicate_rows,
            "deduped_rows": deduped_rows,
            "first_answer_rows_kept": first_answer_rows,
            "followup_rows_kept": followup_rows,
        },
        "duplicate_detection": {
            "same_record_duplicate_groups": duplicate_groups,
            "same_record_duplicate_rows_removed": duplicate_rows,
            "same_record_duplicate_rule": "question_id + user_id + answer_create_time + answer_text_hash_raw 完全一致",
        },
        "template_text_detection": {
            "template_reused_rows": int(base_df["is_template_text_reused"].sum()),
            "template_reused_text_groups": int(len(template_reuse_df)),
            "template_rule": "去重后 answer_text_hash_norm 复用，且跨 question_id 或跨 user_id",
        },
        "top_template_groups": template_reuse_df.head(20).to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "04g_build_base_answer_library",
                "action": "build_traceable_base_answer_library_with_proxy_key",
                "status": "completed",
                "target_path": str(OUTPUT_PATH),
                "details": (
                    f"input={INPUT_PATH.name}; original_rows={original_rows}; "
                    f"deduped_rows={deduped_rows}; duplicate_rows_removed={duplicate_rows}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "04g_build_base_answer_library",
                "action": "write_duplicate_and_template_audits",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"duplicates={DUPLICATE_PATH.name}; templates={TEMPLATE_PATH.name}; "
                    f"template_groups={int(len(template_reuse_df))}"
                ),
            },
        ]
    )

    print(f"Base answer library written to {OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Duplicate audit written to {DUPLICATE_PATH}")
    print(f"Template audit written to {TEMPLATE_PATH}")


if __name__ == "__main__":
    main()
