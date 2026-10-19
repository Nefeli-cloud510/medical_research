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
OUTPUT_CSV_PATH = OUTPUT_TABLE_DIR / "20e_flow_run100_manual_audit_sample_40.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_AUDIT_DIR / "20e_flow_run100_manual_audit_sample_40_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

BINS = [0, 1, 2, 4, 9, 19, 24]
LABELS = ["1", "2", "3-4", "5-9", "10-19", "20-24"]
TARGET_SAMPLE_SIZE = 40
GROUP_MIN_TARGET = 4
TOP_N_PER_TAIL = 2
MID_N_PER_METRIC = 2
TOP_DISAGREEMENT_N = 10

METRIC_SPECS = [
    ("knowledge_differentiation_bge_centroid", "centroid"),
    ("knowledge_differentiation_bge_nearest_prior", "nearest"),
    ("knowledge_differentiation_bge_sentence_mean", "sentence"),
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


def append_cleaning_log(rows: list[dict[str, str]]) -> None:
    file_exists = CLEANING_LOG_PATH.exists()
    is_empty = (not file_exists) or CLEANING_LOG_PATH.stat().st_size == 0
    with CLEANING_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANING_LOG_COLUMNS)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)


def prepare_base_df() -> pd.DataFrame:
    metric_df = pd.read_parquet(INPUT_METRIC_PATH).copy()
    answer_df = pd.read_parquet(INPUT_ANSWER_PATH).copy()

    metric_df["question_id"] = metric_df["question_id"].astype(str)
    answer_df["question_id"] = answer_df["question_id"].astype(str)

    valid_mask = metric_df[
        [
            "knowledge_differentiation_bge_centroid",
            "knowledge_differentiation_bge_nearest_prior",
            "knowledge_differentiation_bge_sentence_mean",
        ]
    ].notna().all(axis=1)

    metric_df = metric_df.loc[valid_mask].copy()
    metric_df["prior_group"] = pd.cut(
        metric_df["peer_crowding_prior_answer_count"],
        bins=BINS,
        labels=LABELS,
    )
    metric_df["metric_spread"] = metric_df[
        [
            "knowledge_differentiation_bge_centroid",
            "knowledge_differentiation_bge_nearest_prior",
            "knowledge_differentiation_bge_sentence_mean",
        ]
    ].max(axis=1) - metric_df[
        [
            "knowledge_differentiation_bge_centroid",
            "knowledge_differentiation_bge_nearest_prior",
            "knowledge_differentiation_bge_sentence_mean",
        ]
    ].min(axis=1)

    merged = metric_df.merge(
        answer_df[
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
    )
    merged["selection_tags"] = ""
    return merged


def append_unique(
    selected: list[str],
    tag_map: dict[str, list[str]],
    candidates: list[str],
    tag: str,
    limit: int | None = None,
) -> None:
    added = 0
    for event_id in candidates:
        if event_id not in tag_map:
            selected.append(event_id)
            tag_map[event_id] = [tag]
            added += 1
        else:
            if tag not in tag_map[event_id]:
                tag_map[event_id].append(tag)
        if limit is not None and added >= limit:
            break


def build_selection(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    selected: list[str] = []
    tag_map: dict[str, list[str]] = {}

    # 1) Each metric: low / mid / high
    for metric_col, metric_label in METRIC_SPECS:
        ordered_low = df.sort_values(
            [metric_col, "metric_spread", "peer_crowding_prior_answer_count", "event_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        ordered_high = df.sort_values(
            [metric_col, "metric_spread", "peer_crowding_prior_answer_count", "event_id"],
            ascending=[False, False, True, True],
            kind="stable",
        )
        median_value = df[metric_col].median()
        ordered_mid = df.assign(
            metric_mid_distance=(df[metric_col] - median_value).abs()
        ).sort_values(
            ["metric_mid_distance", "metric_spread", "peer_crowding_prior_answer_count", "event_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )

        append_unique(
            selected,
            tag_map,
            ordered_low["event_id"].tolist(),
            tag=f"{metric_label}_low",
            limit=TOP_N_PER_TAIL,
        )
        append_unique(
            selected,
            tag_map,
            ordered_mid["event_id"].tolist(),
            tag=f"{metric_label}_mid",
            limit=MID_N_PER_METRIC,
        )
        append_unique(
            selected,
            tag_map,
            ordered_high["event_id"].tolist(),
            tag=f"{metric_label}_high",
            limit=TOP_N_PER_TAIL,
        )

    # 2) Largest disagreements across three metrics
    disagreement_candidates = df.sort_values(
        ["metric_spread", "peer_crowding_prior_answer_count", "event_id"],
        ascending=[False, True, True],
        kind="stable",
    )["event_id"].tolist()
    append_unique(
        selected,
        tag_map,
        disagreement_candidates,
        tag="largest_metric_disagreement",
        limit=TOP_DISAGREEMENT_N,
    )

    # 3) Ensure each prior group has at least GROUP_MIN_TARGET rows if possible
    selected_df = df.loc[df["event_id"].isin(selected)].copy()
    group_counts = Counter(selected_df["prior_group"].astype(str).tolist())
    for group_label in LABELS:
        needed = GROUP_MIN_TARGET - group_counts.get(group_label, 0)
        if needed <= 0:
            continue
        group_candidates = df.loc[df["prior_group"].astype(str) == group_label].sort_values(
            ["metric_spread", "event_id"],
            ascending=[False, True],
            kind="stable",
        )["event_id"].tolist()
        before = len(selected)
        append_unique(
            selected,
            tag_map,
            group_candidates,
            tag=f"group_cover_{group_label}",
            limit=needed,
        )
        group_counts[group_label] += len(selected) - before

    # 4) Fill remaining slots by disagreement, while favoring groups with smaller counts
    while len(selected) < TARGET_SAMPLE_SIZE:
        selected_df = df.loc[df["event_id"].isin(selected)].copy()
        current_group_counts = Counter(selected_df["prior_group"].astype(str).tolist())

        remaining_df = df.loc[~df["event_id"].isin(selected)].copy()
        if remaining_df.empty:
            break
        remaining_df["group_count_current"] = remaining_df["prior_group"].astype(str).map(
            lambda x: current_group_counts.get(x, 0)
        )
        remaining_df = remaining_df.sort_values(
            ["group_count_current", "metric_spread", "peer_crowding_prior_answer_count", "event_id"],
            ascending=[True, False, True, True],
            kind="stable",
        )
        next_event_id = remaining_df.iloc[0]["event_id"]
        append_unique(selected, tag_map, [next_event_id], tag="fill_balance")

    final_df = df.loc[df["event_id"].isin(selected)].copy()
    final_df["selection_tags"] = final_df["event_id"].map(lambda x: "; ".join(tag_map.get(x, [])))
    final_df["prior_group"] = final_df["prior_group"].astype(str)

    final_df = final_df.sort_values(
        ["prior_group", "peer_crowding_prior_answer_count", "metric_spread", "event_id"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    summary = {
        "selected_count": int(len(final_df)),
        "prior_group_counts": final_df["prior_group"].value_counts().sort_index().to_dict(),
        "selection_tag_counts": Counter(tag for tags in tag_map.values() for tag in tags),
    }
    return final_df, summary


def finalize_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["manual_human_score_overall"] = ""
    out["manual_human_score_new_knowledge"] = ""
    out["manual_human_score_repeat_tendency"] = ""
    out["manual_human_notes"] = ""
    out["question_text"] = out["question_desc"].fillna("").astype(str)
    out["all_prior_answers_text"] = out["strict_prior_text_concat"].fillna("").astype(str)
    out["focal_answer_text"] = out["answer_text"].fillna("").astype(str)
    out["sample_note"] = out["prior_group"].map(lambda x: "样本较少，仅作观察" if x == "20-24" else "")

    keep_cols = [
        "question_id",
        "event_id",
        "user_id",
        "time_seq",
        "seq",
        "answer_create_time",
        "prior_group",
        "sample_note",
        "peer_crowding_prior_answer_count",
        "peer_crowding_prior_total_char_count",
        "current_answer_char_count",
        "knowledge_differentiation_bge_centroid",
        "knowledge_differentiation_bge_nearest_prior",
        "knowledge_differentiation_bge_sentence_mean",
        "metric_spread",
        "selection_tags",
        "question_text",
        "all_prior_answers_text",
        "focal_answer_text",
        "strict_prior_event_ids_json",
        "manual_human_score_overall",
        "manual_human_score_new_knowledge",
        "manual_human_score_repeat_tendency",
        "manual_human_notes",
    ]
    out = out[keep_cols].copy()
    out = out.sort_values(
        ["prior_group", "peer_crowding_prior_answer_count", "event_id"],
        kind="stable",
    ).reset_index(drop=True)
    return out


def main() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    base_df = prepare_base_df()
    selected_df, selection_summary = build_selection(base_df)
    output_df = finalize_output(selected_df)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    preview_cols = [
        "question_id",
        "event_id",
        "prior_group",
        "peer_crowding_prior_answer_count",
        "knowledge_differentiation_bge_centroid",
        "knowledge_differentiation_bge_nearest_prior",
        "knowledge_differentiation_bge_sentence_mean",
        "metric_spread",
        "selection_tags",
    ]

    summary = {
        "run_ts_utc": utc_now_iso(),
        "input_metric_path": str(INPUT_METRIC_PATH),
        "input_answer_path": str(INPUT_ANSWER_PATH),
        "output_csv_path": str(OUTPUT_CSV_PATH),
        "target_sample_size": TARGET_SAMPLE_SIZE,
        "actual_sample_size": int(len(output_df)),
        "prior_group_counts": selection_summary["prior_group_counts"],
        "selection_tag_counts": dict(selection_summary["selection_tag_counts"]),
        "preview": output_df[preview_cols].head(10).to_dict(orient="records"),
    }
    OUTPUT_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05j_build_flow_run100_manual_audit_sample_csv",
                "action": "write_manual_audit_sample_csv",
                "status": "completed",
                "target_path": str(OUTPUT_CSV_PATH),
                "details": (
                    f"sample_size={len(output_df)}; "
                    f"prior_groups={summary['prior_group_counts']}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05j_build_flow_run100_manual_audit_sample_csv",
                "action": "write_manual_audit_sample_summary",
                "status": "completed",
                "target_path": str(OUTPUT_SUMMARY_PATH),
                "details": "selection=metric_high_mid_low+largest_disagreement+group_coverage",
            },
        ]
    )

    print(f"Manual audit CSV written to {OUTPUT_CSV_PATH}")
    print(f"Summary written to {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
