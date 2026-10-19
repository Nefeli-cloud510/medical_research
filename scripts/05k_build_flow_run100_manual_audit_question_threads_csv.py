from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
OUTPUT_TABLE_DIR = ROOT_DIR / "outputs" / "tables"
OUTPUT_AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

INPUT_METRIC_PATH = ANALYSIS_DIR / "flow_run100_knowledge_metrics_bge.parquet"
INPUT_ANSWER_PATH = ANALYSIS_DIR / "flow_run100_answer_text_features.parquet"
OUTPUT_CSV_PATH = OUTPUT_TABLE_DIR / "20f_flow_run100_manual_audit_question_threads_10.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_AUDIT_DIR / "20f_flow_run100_manual_audit_question_threads_10_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

TARGET_QUESTION_COUNT = 10

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


def append_cleaning_log(rows: list[dict[str, str]]) -> None:
    file_exists = CLEANING_LOG_PATH.exists()
    is_empty = (not file_exists) or CLEANING_LOG_PATH.stat().st_size == 0
    with CLEANING_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANING_LOG_COLUMNS)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)


def prepare_question_summary(metric_df: pd.DataFrame) -> pd.DataFrame:
    work_df = metric_df.copy()
    work_df["valid_all_three"] = work_df[
        [
            "knowledge_differentiation_bge_centroid",
            "knowledge_differentiation_bge_nearest_prior",
            "knowledge_differentiation_bge_sentence_mean",
        ]
    ].notna().all(axis=1)

    summary = (
        work_df.groupby("question_id", sort=False)
        .agg(
            answer_rows=("event_id", "count"),
            valid_rows=("valid_all_three", "sum"),
            centroid_mean=("knowledge_differentiation_bge_centroid", "mean"),
            nearest_mean=("knowledge_differentiation_bge_nearest_prior", "mean"),
            sentence_mean=("knowledge_differentiation_bge_sentence_mean", "mean"),
            centroid_median=("knowledge_differentiation_bge_centroid", "median"),
            nearest_median=("knowledge_differentiation_bge_nearest_prior", "median"),
            sentence_median=("knowledge_differentiation_bge_sentence_mean", "median"),
        )
        .reset_index()
    )
    summary["question_metric_spread_mean"] = summary[
        ["centroid_mean", "nearest_mean", "sentence_mean"]
    ].max(axis=1) - summary[
        ["centroid_mean", "nearest_mean", "sentence_mean"]
    ].min(axis=1)
    return summary


def append_unique_question(
    selected: list[str],
    tag_map: dict[str, list[str]],
    question_ids: list[str],
    tag: str,
    limit: int,
) -> None:
    added = 0
    for question_id in question_ids:
        if question_id not in tag_map:
            selected.append(question_id)
            tag_map[question_id] = [tag]
            added += 1
        elif tag not in tag_map[question_id]:
            tag_map[question_id].append(tag)
        if added >= limit:
            break


def build_question_selection(summary_df: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    selected: list[str] = []
    tag_map: dict[str, list[str]] = {}

    ordered = summary_df.sort_values(
        ["centroid_mean", "question_metric_spread_mean", "answer_rows", "question_id"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    append_unique_question(selected, tag_map, ordered["question_id"].tolist(), "question_centroid_low", 2)

    ordered = summary_df.assign(
        centroid_mid_distance=(summary_df["centroid_mean"] - summary_df["centroid_mean"].median()).abs()
    ).sort_values(
        ["centroid_mid_distance", "question_metric_spread_mean", "answer_rows", "question_id"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    append_unique_question(selected, tag_map, ordered["question_id"].tolist(), "question_centroid_mid", 2)

    ordered = summary_df.sort_values(
        ["centroid_mean", "question_metric_spread_mean", "answer_rows", "question_id"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    append_unique_question(selected, tag_map, ordered["question_id"].tolist(), "question_centroid_high", 2)

    ordered = summary_df.sort_values(
        ["question_metric_spread_mean", "answer_rows", "question_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    append_unique_question(selected, tag_map, ordered["question_id"].tolist(), "question_metric_disagreement_high", 2)

    ordered = summary_df.sort_values(
        ["answer_rows", "question_metric_spread_mean", "question_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    append_unique_question(selected, tag_map, ordered["question_id"].tolist(), "question_dense_thread", 1)

    ordered = summary_df.sort_values(
        ["answer_rows", "question_metric_spread_mean", "question_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    append_unique_question(selected, tag_map, ordered["question_id"].tolist(), "question_sparse_thread", 1)

    if len(selected) < TARGET_QUESTION_COUNT:
        ordered = summary_df.sort_values(
            ["question_metric_spread_mean", "centroid_mean", "answer_rows", "question_id"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        append_unique_question(
            selected,
            tag_map,
            ordered["question_id"].tolist(),
            "question_fill_balance",
            TARGET_QUESTION_COUNT - len(selected),
        )

    return selected[:TARGET_QUESTION_COUNT], tag_map


def build_output(
    question_ids: list[str],
    question_tag_map: dict[str, list[str]],
    metric_df: pd.DataFrame,
    answer_df: pd.DataFrame,
    question_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    selected_metric_df = metric_df.loc[metric_df["question_id"].isin(question_ids)].copy()
    selected_answer_df = answer_df.loc[answer_df["question_id"].isin(question_ids)].copy()
    question_rank = {question_id: idx + 1 for idx, question_id in enumerate(question_ids)}

    output_df = selected_metric_df.merge(
        selected_answer_df[
            [
                "event_id",
                "question_id",
                "question_desc",
                "answer_text",
                "strict_prior_text_concat",
                "strict_prior_event_ids_json",
            ]
        ],
        on=["event_id", "question_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        question_summary_df,
        on="question_id",
        how="left",
        validate="many_to_one",
    )

    output_df["question_sample_order"] = output_df["question_id"].map(question_rank)
    output_df["question_selection_tags"] = output_df["question_id"].map(
        lambda x: "; ".join(question_tag_map.get(x, []))
    )
    output_df["question_text"] = output_df["question_desc"].fillna("").astype(str)
    output_df["all_prior_answers_text"] = output_df["strict_prior_text_concat"].fillna("").astype(str)
    output_df["focal_answer_text"] = output_df["answer_text"].fillna("").astype(str)
    output_df["manual_human_score_overall"] = ""
    output_df["manual_human_score_new_knowledge"] = ""
    output_df["manual_human_score_repeat_tendency"] = ""
    output_df["manual_human_notes"] = ""

    keep_cols = [
        "question_sample_order",
        "question_selection_tags",
        "question_id",
        "answer_rows",
        "valid_rows",
        "question_metric_spread_mean",
        "centroid_mean",
        "nearest_mean",
        "sentence_mean",
        "event_id",
        "user_id",
        "time_seq",
        "seq",
        "answer_create_time",
        "peer_crowding_prior_answer_count",
        "peer_crowding_prior_total_char_count",
        "current_answer_char_count",
        "knowledge_differentiation_bge_centroid",
        "knowledge_differentiation_bge_nearest_prior",
        "knowledge_differentiation_bge_sentence_mean",
        "question_text",
        "all_prior_answers_text",
        "focal_answer_text",
        "strict_prior_event_ids_json",
        "manual_human_score_overall",
        "manual_human_score_new_knowledge",
        "manual_human_score_repeat_tendency",
        "manual_human_notes",
    ]
    output_df = output_df[keep_cols].copy()
    output_df = output_df.sort_values(
        ["question_sample_order", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    return output_df


def main() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    metric_df = pd.read_parquet(INPUT_METRIC_PATH).copy()
    answer_df = pd.read_parquet(INPUT_ANSWER_PATH).copy()
    metric_df["question_id"] = metric_df["question_id"].astype(str)
    answer_df["question_id"] = answer_df["question_id"].astype(str)

    question_summary_df = prepare_question_summary(metric_df)
    selected_question_ids, question_tag_map = build_question_selection(question_summary_df)
    output_df = build_output(
        question_ids=selected_question_ids,
        question_tag_map=question_tag_map,
        metric_df=metric_df,
        answer_df=answer_df,
        question_summary_df=question_summary_df,
    )
    output_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    question_preview = question_summary_df.loc[
        question_summary_df["question_id"].isin(selected_question_ids)
    ].copy()
    question_preview["question_selection_tags"] = question_preview["question_id"].map(
        lambda x: "; ".join(question_tag_map.get(x, []))
    )
    question_preview["question_sample_order"] = question_preview["question_id"].map(
        {question_id: idx + 1 for idx, question_id in enumerate(selected_question_ids)}
    )
    question_preview = question_preview.sort_values("question_sample_order", kind="stable")

    summary = {
        "run_ts_utc": utc_now_iso(),
        "input_metric_path": str(INPUT_METRIC_PATH),
        "input_answer_path": str(INPUT_ANSWER_PATH),
        "output_csv_path": str(OUTPUT_CSV_PATH),
        "selected_question_count": len(selected_question_ids),
        "selected_answer_row_count": int(len(output_df)),
        "question_tag_counts": dict(Counter(tag for tags in question_tag_map.values() for tag in tags)),
        "selected_questions": question_preview.to_dict(orient="records"),
    }
    OUTPUT_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05k_build_flow_run100_manual_audit_question_threads_csv",
                "action": "write_question_thread_manual_audit_csv",
                "status": "completed",
                "target_path": str(OUTPUT_CSV_PATH),
                "details": (
                    f"question_count={len(selected_question_ids)}; "
                    f"answer_rows={len(output_df)}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05k_build_flow_run100_manual_audit_question_threads_csv",
                "action": "write_question_thread_manual_audit_summary",
                "status": "completed",
                "target_path": str(OUTPUT_SUMMARY_PATH),
                "details": "selection=question_level_centroid_high_mid_low+disagreement+dense_sparse",
            },
        ]
    )

    print(f"Question-thread audit CSV written to {OUTPUT_CSV_PATH}")
    print(f"Summary written to {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
