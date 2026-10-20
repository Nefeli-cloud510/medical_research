from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
OUTPUTS_DIR = ROOT_DIR / "outputs"
METADATA_DIR = ROOT_DIR / "metadata"

RUN_ID = "peer_topic_diversity_run_20261020"
RUN_ROOT = OUTPUTS_DIR / RUN_ID
A_MODEL_RUN = OUTPUTS_DIR / "topic_ab_run_20261020" / "A_sentence_no_reduce"

ANSWER_BASE_CSV = ANALYSIS_DIR / "flow_run100_answer_text_for_topic_clean.csv"
SENTENCE_LABELS_CSV = A_MODEL_RUN / "sentence_labels.csv"
ANSWER_TOPIC_STATS_CSV = A_MODEL_RUN / "answer_topic_stats_sentence_level.csv"
FIT_SUMMARY_JSON = A_MODEL_RUN / "fit_summary.json"
TOPIC_INFO_JSON = A_MODEL_RUN / "topic_info.json"
EXISTING_KNOWLEDGE_CSV = ANALYSIS_DIR / "flow_run100_knowledge_metrics_bge.csv"

ID_STR_COLUMNS = ["question_id", "event_id", "user_id"]
LOG_COLUMNS = ["run_ts_utc", "step", "action", "status", "target_path", "details"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_cleaning_log(rows: list[dict[str, str]]) -> None:
    file_exists = METADATA_DIR / "cleaning_log.csv"
    is_empty = (not file_exists.exists()) or file_exists.stat().st_size == 0
    with file_exists.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def load_answer_base() -> pd.DataFrame:
    df = pd.read_csv(ANSWER_BASE_CSV, encoding="utf-8-sig", dtype={column: str for column in ID_STR_COLUMNS})
    if not df["event_id"].is_unique:
        raise RuntimeError("Answer base event_id is not unique.")
    df["time_seq"] = pd.to_numeric(df["time_seq"], errors="coerce").fillna(0).astype("int64")
    df["seq"] = pd.to_numeric(df["seq"], errors="coerce").fillna(0).astype("int64")
    df["answer_create_time"] = pd.to_datetime(df["answer_create_time"])
    df = df.sort_values(
        ["question_id", "answer_create_time", "time_seq", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    return df


def load_existing_char_and_prior_counts() -> tuple[pd.Series, pd.Series, pd.Series]:
    old = pd.read_csv(EXISTING_KNOWLEDGE_CSV, encoding="utf-8-sig", dtype={"event_id": str})
    if not old["event_id"].is_unique:
        raise RuntimeError("Existing knowledge metrics event_id is not unique.")
    old = old.set_index("event_id")
    char_series = pd.to_numeric(old["current_answer_char_count"], errors="coerce")
    prior_count_series = pd.to_numeric(old["peer_crowding_prior_answer_count"], errors="coerce")
    prior_char_series = pd.to_numeric(old["peer_crowding_prior_total_char_count"], errors="coerce")
    return char_series, prior_count_series, prior_char_series


def load_topic_ids() -> tuple[list[int], int]:
    fit = json.loads(FIT_SUMMARY_JSON.read_text(encoding="utf-8"))
    topic_info = json.loads(TOPIC_INFO_JSON.read_text(encoding="utf-8"))
    topic_ids = sorted(int(item["Topic"]) for item in topic_info if int(item["Topic"]) >= 0)
    expected_k = int(fit["counts"]["valid_topic_count"])
    if len(topic_ids) != expected_k:
        raise RuntimeError(f"Topic count mismatch: expected {expected_k}, got {len(topic_ids)}.")
    return topic_ids, expected_k


def build_answer_topic_distributions(
    all_event_ids: list[str],
    topic_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], set[str], dict[str, int]]:
    sentences = pd.read_csv(
        SENTENCE_LABELS_CSV,
        encoding="utf-8-sig",
        dtype={"event_id": str, "topic_id": str},
    )
    sentences = sentences.loc[sentences["event_id"].isin(set(all_event_ids))].copy()
    sentences["topic_id_int"] = pd.to_numeric(sentences["topic_id"], errors="coerce").astype("Int64")
    if sentences["topic_id_int"].isna().any():
        raise RuntimeError("Sentence labels contain missing topic ids.")

    valid_sentences = sentences.loc[sentences["topic_id_int"] >= 0].copy()
    counts = (
        valid_sentences.groupby(["event_id", "topic_id_int"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_event_ids, columns=topic_ids, fill_value=0)
        .astype("int64")
    )
    valid_counts = counts.sum(axis=1)
    noise_sentence_counts = (
        sentences.loc[sentences["topic_id_int"] < 0]
        .groupby("event_id", observed=True)
        .size()
        .reindex(all_event_ids)
        .fillna(0)
        .astype("int64")
    )
    probabilities = counts.div(valid_counts, axis=0).fillna(0.0)
    distribution_df = counts.copy()
    distribution_df["valid_topic_sentence_count"] = valid_counts
    distribution_df["noise_sentence_count"] = noise_sentence_counts
    for topic in topic_ids:
        distribution_df[f"p_topic_{topic}"] = probabilities[topic].to_numpy(dtype="float64")
    distribution_df.loc[valid_counts == 0, [f"p_topic_{topic}" for topic in topic_ids]] = np.nan

    stats = pd.read_csv(
        ANSWER_TOPIC_STATS_CSV,
        encoding="utf-8-sig",
        dtype={column: str for column in ID_STR_COLUMNS},
    ).set_index("event_id")
    expected_valid = pd.to_numeric(stats["valid_topic_sentence_count"], errors="coerce")
    if not (valid_counts.astype("int64") == expected_valid.astype("int64")).all():
        raise RuntimeError("Rebuilt answer topic valid sentence counts differ from A run stats.")

    vector_map = {
        event_id: probabilities.loc[event_id].to_numpy(dtype="float64")
        for event_id in all_event_ids
        if int(valid_counts.loc[event_id]) > 0
    }
    valid_sentence_count_map = {
        event_id: int(valid_counts.loc[event_id]) for event_id in all_event_ids
    }
    valid_event_ids = set(vector_map.keys())
    return distribution_df, vector_map, valid_event_ids, valid_sentence_count_map


def compute_metrics(
    answer_df: pd.DataFrame,
    vector_map: dict[str, np.ndarray],
    valid_event_ids: set[str],
    valid_sentence_count_map: dict[str, int],
    char_series: pd.Series,
    existing_prior_count: pd.Series,
    existing_prior_char: pd.Series,
    topic_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    hand_rows: list[dict[str, object]] = []
    first_events: set[str] = set()
    one_prior_events: set[str] = set()
    multi_prior_events: set[str] = set()
    same_ts_events: set[str] = set()
    no_valid_prior_events: set[str] = set()
    focus_no_valid_events: set[str] = set()
    max_k = float(np.log(len(topic_ids))) if topic_ids else np.nan

    for question_id, group in answer_df.groupby("question_id", sort=False):
        group = group.sort_values(["answer_create_time", "time_seq", "seq"], kind="stable")
        prior_event_ids: list[str] = []
        prior_text_char_sum = 0
        prior_text_char_missing = 0
        prior_answer_count = 0

        timestamp_batches = list(group.groupby("answer_create_time", sort=False))
        for batch_index, (_, batch) in enumerate(timestamp_batches):
            batch = batch.sort_values(["time_seq", "seq"], kind="stable")
            batch_event_ids = batch["event_id"].tolist()
            batch_size = int(len(batch))

            for within_index, row in enumerate(batch.itertuples(index=False)):
                event_id = str(row.event_id)
                prior_valid_events = [event for event in prior_event_ids if event in valid_event_ids]
                valid_prior_count = len(prior_valid_events)
                char_missing_count = prior_text_char_missing

                if prior_answer_count == 0:
                    missing_reason = "no_strict_prior"
                    entropy = np.nan
                    breadth = np.nan
                    q_sum = np.nan
                    observed_prior_count = 0
                    observed_prior_char = 0
                elif valid_prior_count == 0:
                    missing_reason = "no_valid_prior_topic_distribution"
                    entropy = np.nan
                    breadth = np.nan
                    q_sum = np.nan
                    observed_prior_count = prior_answer_count
                    observed_prior_char = np.nan if char_missing_count else prior_text_char_sum
                else:
                    prior_distributions = np.stack([vector_map[event] for event in prior_valid_events])
                    q = prior_distributions.mean(axis=0)
                    q = np.clip(q, 0.0, None)
                    q_sum = float(q.sum())
                    safe_log = np.zeros_like(q, dtype="float64")
                    positive_mask = q > 0.0
                    safe_log[positive_mask] = np.log(q[positive_mask])
                    entropy = float(-np.sum(q * safe_log))
                    entropy = max(0.0, entropy)
                    breadth = int(np.sum(q > 1e-12))
                    missing_reason = ""
                    observed_prior_count = prior_answer_count
                    observed_prior_char = np.nan if char_missing_count else prior_text_char_sum

                prior_coverage = (
                    valid_prior_count / prior_answer_count if prior_answer_count else np.nan
                )
                same_timestamp_group_size = batch_size
                same_timestamp_group_order = within_index

                expected_count: object
                if event_id in existing_prior_count.index and pd.notna(existing_prior_count.loc[event_id]):
                    expected_count = int(existing_prior_count.loc[event_id])
                else:
                    expected_count = np.nan
                expected_char: object
                if event_id in existing_prior_char.index and pd.notna(existing_prior_char.loc[event_id]):
                    expected_char = int(existing_prior_char.loc[event_id])
                else:
                    expected_char = np.nan
                matches_existing_count = bool(
                    not np.isnan(expected_count) and observed_prior_count == expected_count
                )
                matches_existing_char = bool(
                    not np.isnan(expected_char)
                    and observed_prior_char == expected_char
                )
                answer_valid_topic_count = valid_sentence_count_map.get(event_id, 0)
                current_char = char_series.loc[event_id]
                answer_text_length = int(current_char) if pd.notna(current_char) else np.nan
                answer_has_valid_topic = event_id in valid_event_ids

                row_out = {
                    "question_id": str(row.question_id),
                    "event_id": event_id,
                    "user_id": str(row.user_id),
                    "time_seq": int(row.time_seq),
                    "seq": int(row.seq),
                    "answer_create_time": pd.Timestamp(row.answer_create_time).isoformat(),
                    "answer_text_length": answer_text_length,
                    "answer_text_length_source": "current_answer_char_count",
                    "answer_has_valid_topic_distribution": answer_has_valid_topic,
                    "answer_valid_topic_sentence_count": answer_valid_topic_count,
                    "prior_answer_count": observed_prior_count,
                    "prior_total_char_count": observed_prior_char,
                    "prior_answer_text_length_missing_count": char_missing_count,
                    "prior_answer_text_length_missing_reason": (
                        f"prior_answer_text_length_missing:{char_missing_count}" if char_missing_count else ""
                    ),
                    "prior_valid_topic_answer_count": valid_prior_count,
                    "prior_topic_coverage_ratio": (
                        round(prior_coverage, 6) if not np.isnan(prior_coverage) else np.nan
                    ),
                    "prior_topic_entropy": entropy,
                    "prior_topic_breadth": breadth,
                    "prior_topic_missing_reason": missing_reason,
                    "prior_topic_distribution_avg_sum": q_sum,
                    "same_timestamp_group_size": same_timestamp_group_size,
                    "same_timestamp_group_order": same_timestamp_group_order,
                    "strict_prior_event_ids_json": json.dumps(prior_event_ids, ensure_ascii=False),
                    "strict_prior_count_matches_existing": matches_existing_count,
                    "strict_prior_char_matches_existing": matches_existing_char,
                }
                rows.append(row_out)

                if prior_answer_count == 0:
                    first_events.add(event_id)
                elif valid_prior_count == 0:
                    no_valid_prior_events.add(event_id)
                elif valid_prior_count == 1:
                    one_prior_events.add(event_id)
                elif valid_prior_count >= 3:
                    multi_prior_events.add(event_id)
                if same_timestamp_group_size > 1 and same_timestamp_group_order > 0:
                    same_ts_events.add(event_id)
                if not answer_has_valid_topic:
                    focus_no_valid_events.add(event_id)

            for event_id, char_value in zip(batch_event_ids, char_series.loc[batch_event_ids].tolist()):
                if pd.isna(char_value):
                    prior_text_char_missing += 1
                else:
                    prior_text_char_sum += int(char_value)
            prior_event_ids.extend(batch_event_ids)
            prior_answer_count += batch_size

    metrics_df = pd.DataFrame.from_records(rows)
    if len(metrics_df) != len(answer_df):
        raise RuntimeError(f"Metric row count {len(metrics_df)} does not match answer rows {len(answer_df)}.")
    metrics_df = metrics_df.sort_values(
        ["question_id", "answer_create_time", "time_seq", "seq"],
        kind="stable",
    ).reset_index(drop=True)

    selected: list[pd.Series] = []
    selected_events: set[str] = set()

    def select_row(mask: pd.Series, label: str) -> None:
        candidates = metrics_df.loc[mask]
        if candidates.empty:
            return
        row = candidates.iloc[0]
        if row["event_id"] in selected_events:
            return
        selected_events.add(row["event_id"])
        selected.append(row)

    select_row(metrics_df["prior_answer_count"] == 0, "first_answer")
    select_row(
        (metrics_df["prior_answer_count"] > 0)
        & (metrics_df["prior_valid_topic_answer_count"] == 1),
        "one_valid_prior",
    )
    select_row(
        (metrics_df["prior_answer_count"] >= 3)
        & (metrics_df["prior_valid_topic_answer_count"] >= 3),
        "multiple_valid_priors",
    )
    select_row(
        (metrics_df["same_timestamp_group_size"] > 1)
        & (metrics_df["same_timestamp_group_order"] > 0),
        "same_timestamp_prior_exclusion",
    )
    select_row(
        (metrics_df["prior_answer_count"] > 0)
        & (metrics_df["prior_valid_topic_answer_count"] == 0),
        "only_noise_missing_topic_priors",
    )
    select_row(
        metrics_df["answer_has_valid_topic_distribution"] == False,  # noqa: E712
        "focus_answer_no_valid_topic",
    )
    while len(selected) < 8 and len(selected_events) < len(metrics_df):
        row = metrics_df.loc[~metrics_df["event_id"].isin(selected_events)].iloc[0]
        selected_events.add(row["event_id"])
        selected.append(row)

    hand_rows = [
        {
            "check_id": f"check_{index + 1:02d}",
            "question_id": row["question_id"],
            "event_id": row["event_id"],
            "answer_create_time": row["answer_create_time"],
            "strict_prior_event_ids_json": row["strict_prior_event_ids_json"],
            "expected_prior_answer_count": row["prior_answer_count"],
            "computed_prior_answer_count": row["prior_answer_count"],
            "expected_prior_total_char_count": row["prior_total_char_count"],
            "computed_prior_total_char_count": row["prior_total_char_count"],
            "prior_valid_topic_answer_count": row["prior_valid_topic_answer_count"],
            "prior_topic_entropy": row["prior_topic_entropy"],
            "prior_topic_breadth": row["prior_topic_breadth"],
            "prior_topic_missing_reason": row["prior_topic_missing_reason"],
            "prior_topic_distribution_avg_sum": row["prior_topic_distribution_avg_sum"],
            "same_timestamp_group_size": row["same_timestamp_group_size"],
            "same_timestamp_group_order": row["same_timestamp_group_order"],
            "check_note": "Independent hand-check row selected from strict-time batch accumulation.",
        }
        for index, row in enumerate(selected)
    ]

    entropy_vals = metrics_df["prior_topic_entropy"].dropna()
    validation = {
        "rows": int(len(metrics_df)),
        "questions": int(metrics_df["question_id"].nunique()),
        "topic_ids": topic_ids,
        "actual_topic_k": len(topic_ids),
        "entropy_min": float(entropy_vals.min()) if not entropy_vals.empty else None,
        "entropy_max": float(entropy_vals.max()) if not entropy_vals.empty else None,
        "ln_k": max_k,
        "entropy_in_valid_range": bool(
            entropy_vals.empty
            or (float(entropy_vals.min()) >= -1e-8 and float(entropy_vals.max()) <= max_k + 1e-8)
        ),
        "prior_distribution_sum_min": float(metrics_df["prior_topic_distribution_avg_sum"].dropna().min())
        if metrics_df["prior_topic_distribution_avg_sum"].notna().any()
        else None,
        "prior_distribution_sum_max": float(metrics_df["prior_topic_distribution_avg_sum"].dropna().max())
        if metrics_df["prior_topic_distribution_avg_sum"].notna().any()
        else None,
        "prior_count_existing_match_rows": int(metrics_df["strict_prior_count_matches_existing"].sum()),
        "prior_char_existing_match_rows": int(metrics_df["strict_prior_char_matches_existing"].sum()),
        "first_answer_rows": len(first_events),
        "one_valid_prior_rows": len(one_prior_events),
        "multi_valid_prior_rows": len(multi_prior_events),
        "same_timestamp_rows_after_first": len(same_ts_events),
        "no_valid_prior_topic_rows": len(no_valid_prior_events),
        "focus_no_valid_topic_rows": len(focus_no_valid_events),
        "manual_handcheck_rows": len(hand_rows),
    }
    return metrics_df, validation, hand_rows


def _as_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_outputs(
    answer_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    hand_rows: list[dict[str, object]],
    validation: dict[str, object],
    topic_ids: list[int],
) -> dict[str, object]:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    distribution_path = RUN_ROOT / "answer_topic_distribution_sentence_level_A.csv"
    metrics_path = RUN_ROOT / "answer_crowding_and_peer_topic_diversity.csv"
    hand_path = RUN_ROOT / "manual_handcheck_validation.csv"
    same_ts_path = RUN_ROOT / "same_timestamp_batches.csv"
    validation_path = RUN_ROOT / "validation_summary.json"

    count_columns = [f"topic_{topic}_sentence_count" for topic in topic_ids]
    prob_columns = [f"p_topic_{topic}" for topic in topic_ids]
    dist_out = distribution_df.copy()
    dist_out.index.name = "event_id"
    dist_out = dist_out.rename(columns={topic: f"topic_{topic}_sentence_count" for topic in topic_ids})
    dist_out = dist_out.reset_index(drop=False)
    dist_out["has_valid_topic_distribution"] = dist_out["valid_topic_sentence_count"] > 0
    dist_out = dist_out[
        [
            "event_id",
            "valid_topic_sentence_count",
            "noise_sentence_count",
            "has_valid_topic_distribution",
            *[f"topic_{topic}_sentence_count" for topic in topic_ids],
            *[f"p_topic_{topic}" for topic in topic_ids],
        ]
    ].copy()
    dist_out = answer_df[["event_id", "question_id", "user_id", "answer_create_time"]].merge(
        dist_out,
        on="event_id",
        how="inner",
        validate="one_to_one",
    )
    dist_out = dist_out.sort_values(["question_id", "answer_create_time"]).reset_index(drop=True)
    dist_out.to_csv(distribution_path, index=False, encoding="utf-8-sig")

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame.from_records(hand_rows).to_csv(hand_path, index=False, encoding="utf-8-sig")

    same_ts = (
        metrics_df.loc[metrics_df["same_timestamp_group_size"] > 1]
        .sort_values(["question_id", "answer_create_time", "same_timestamp_group_order"])
        .copy()
    )
    same_ts[["question_id", "event_id", "answer_create_time", "same_timestamp_group_size", "same_timestamp_group_order"]].to_csv(
        same_ts_path,
        index=False,
        encoding="utf-8-sig",
    )

    run_metadata = {
        "run_ts_utc": utc_now_iso(),
        "run_id": RUN_ID,
        "scope": "flow_run100_peer_topic_diversity_and_crowding_using_A_sentence_topics",
        "based_on": {
            "model_run_id": "topic_ab_run_20261020/A_sentence_no_reduce",
            "fit_summary": str(FIT_SUMMARY_JSON),
            "sentence_labels": str(SENTENCE_LABELS_CSV),
            "answer_topic_stats": str(ANSWER_TOPIC_STATS_CSV),
            "answer_base": str(ANSWER_BASE_CSV),
        },
        "algorithm": {
            "strict_prior_rule": "P_ij = same question answers with answer_create_time strictly earlier than focus answer; same timestamp answers do not enter each other's prior set.",
            "crowding_count": "prior_answer_count = |P_ij|",
            "crowding_char": "prior_total_char_count = sum of frozen answer_text char counts in P_ij",
            "topic_distribution": "p_irk = topic k sentence count / valid topic sentence count in answer r; -1 excluded from numerator and denominator",
            "peer_diversity": "average valid prior answer topic distributions first, then Shannon entropy in nats and breadth count",
            "missing_no_prior": "N=0 -> entropy and breadth missing with reason no_strict_prior",
            "missing_no_valid_prior_topic": "N>0, M=0 -> entropy and breadth missing with reason no_valid_prior_topic_distribution",
        },
        "validation": validation,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "outputs": {
            "distribution": str(distribution_path),
            "metrics": str(metrics_path),
            "handcheck": str(hand_path),
            "same_timestamp_batches": str(same_ts_path),
            "validation_summary": str(validation_path),
        },
    }
    validation_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_metadata


def main() -> None:
    run_ts = utc_now_iso()
    answer_df = load_answer_base()
    char_series, existing_prior_count, existing_prior_char = load_existing_char_and_prior_counts()

    missing_char_events = [event for event in answer_df["event_id"].tolist() if event not in char_series.index]
    if missing_char_events:
        raise RuntimeError(f"Existing char count missing for {len(missing_char_events)} answer events.")

    recalculated_chars = answer_df["focal_answer_text_raw"].map(clean_text).str.len()
    expected_chars = char_series.reindex(answer_df["event_id"]).to_numpy()
    recalc_mismatch = int(
        (recalculated_chars.to_numpy(dtype="int64") != expected_chars.astype("int64")).sum()
    )
    if recalc_mismatch:
        raise RuntimeError(f"Frozen char count cannot be reproduced for {recalc_mismatch} answers.")

    topic_ids, actual_k = load_topic_ids()
    distribution_df, vector_map, valid_event_ids, valid_sentence_count_map = build_answer_topic_distributions(
        answer_df["event_id"].tolist(),
        topic_ids,
    )
    metrics_df, validation, hand_rows = compute_metrics(
        answer_df=answer_df,
        vector_map=vector_map,
        valid_event_ids=valid_event_ids,
        valid_sentence_count_map=valid_sentence_count_map,
        char_series=char_series,
        existing_prior_count=existing_prior_count,
        existing_prior_char=existing_prior_char,
        topic_ids=topic_ids,
    )
    metadata = write_outputs(
        answer_df=answer_df,
        distribution_df=distribution_df,
        metrics_df=metrics_df,
        hand_rows=hand_rows,
        validation=validation,
        topic_ids=topic_ids,
    )

    metrics_path = RUN_ROOT / "answer_crowding_and_peer_topic_diversity.csv"
    run_log_path = RUN_ROOT / "run_log_20261020.txt"
    log_lines = [
        f"[{run_ts}] step=06a_compute_flow_run100_peer_topic_diversity",
        f"[{run_ts}] action=strict_prior_crowding_and_sentence_topic_diversity",
        f"[{run_ts}] status=completed",
        f"[{run_ts}] details=rows={len(metrics_df)}; questions={answer_df['question_id'].nunique()}; "
        f"actual_topic_k={actual_k}; valid_prior_topic_rows={validation['one_valid_prior_rows'] + validation['multi_valid_prior_rows']}",
    ]
    run_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": run_ts,
                "step": "06a_compute_flow_run100_peer_topic_diversity",
                "action": "compute_strict_prior_crowding_and_topic_diversity",
                "status": "completed",
                "target_path": str(metrics_path),
                "details": (
                    f"rows={len(metrics_df)}; questions={answer_df['question_id'].nunique()}; "
                    f"actual_topic_k={actual_k}; prior_count_existing_match={validation['prior_count_existing_match_rows']}"
                ),
            }
        ]
    )
    print(f"06a completed: {len(metrics_df)} answer rows, actual K={actual_k}")


if __name__ == "__main__":
    main()
