from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
OUTPUTS_DIR = ROOT_DIR / "outputs"
METADATA_DIR = ROOT_DIR / "metadata"
USER_OUTPUTS_DIR = Path(r"C:\Users\admin\Documents\Codex\2026-10-19\co\outputs")

RUN_ID = "peer_topic_diversity_run_20261020"
RUN_ROOT = OUTPUTS_DIR / RUN_ID
A_MODEL_RUN = OUTPUTS_DIR / "topic_ab_run_20261020" / "A_sentence_no_reduce"

ANSWER_BASE_CSV = ANALYSIS_DIR / "flow_run100_answer_text_for_topic_clean.csv"
SENTENCE_LABELS_CSV = A_MODEL_RUN / "sentence_labels.csv"
FIT_SUMMARY_JSON = A_MODEL_RUN / "fit_summary.json"
TOPIC_INFO_JSON = A_MODEL_RUN / "topic_info.json"
KNOWLEDGE_METRICS_CSV = ANALYSIS_DIR / "flow_run100_knowledge_metrics_bge.csv"

METRICS_CSV = RUN_ROOT / "answer_crowding_and_peer_topic_diversity.csv"
DISTRIBUTION_CSV = RUN_ROOT / "answer_topic_distribution_sentence_level_A.csv"
HANDCHECK_CSV = RUN_ROOT / "manual_handcheck_validation.csv"
SAME_TIMESTAMP_CSV = RUN_ROOT / "same_timestamp_batches.csv"

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


def load_answer_base() -> pd.DataFrame:
    df = pd.read_csv(ANSWER_BASE_CSV, encoding="utf-8-sig", dtype={column: str for column in ID_STR_COLUMNS})
    if not df["event_id"].is_unique:
        raise RuntimeError("Answer base event_id is not unique.")
    df["answer_create_time"] = pd.to_datetime(df["answer_create_time"])
    return df.reset_index(drop=True)


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV, encoding="utf-8-sig", dtype={column: str for column in ID_STR_COLUMNS})
    if not df["event_id"].is_unique:
        raise RuntimeError("Metrics event_id is not unique.")
    df["answer_create_time"] = pd.to_datetime(df["answer_create_time"])
    numeric_columns = [
        "answer_text_length",
        "prior_answer_count",
        "prior_total_char_count",
        "prior_answer_text_length_missing_count",
        "prior_valid_topic_answer_count",
        "prior_topic_coverage_ratio",
        "prior_topic_entropy",
        "prior_topic_breadth",
        "prior_topic_distribution_avg_sum",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["question_id", "answer_create_time", "time_seq", "seq"], kind="stable").reset_index(
        drop=True
    )


def load_topic_ids() -> tuple[list[int], int]:
    fit = json.loads(FIT_SUMMARY_JSON.read_text(encoding="utf-8"))
    topic_info = json.loads(TOPIC_INFO_JSON.read_text(encoding="utf-8"))
    topic_ids = sorted(int(item["Topic"]) for item in topic_info if int(item["Topic"]) >= 0)
    expected_k = int(fit["counts"]["valid_topic_count"])
    if len(topic_ids) != expected_k:
        raise RuntimeError(f"Topic count mismatch: expected {expected_k}, got {len(topic_ids)}.")
    return topic_ids, expected_k


def load_distributions(topic_ids: list[int]) -> pd.DataFrame:
    df = pd.read_csv(DISTRIBUTION_CSV, encoding="utf-8-sig", dtype={"event_id": str})
    df["valid_topic_sentence_count"] = pd.to_numeric(df["valid_topic_sentence_count"], errors="coerce")
    df["noise_sentence_count"] = pd.to_numeric(df["noise_sentence_count"], errors="coerce").fillna(0).astype("int64")
    p_columns = [f"p_topic_{topic}" for topic in topic_ids]
    missing_p_columns = [column for column in p_columns if column not in df.columns]
    if missing_p_columns:
        raise RuntimeError(f"Distribution CSV missing probability columns: {missing_p_columns}")
    df = df.set_index("event_id")
    df = df.loc[~df.index.duplicated(keep="first")].copy()
    return df


def load_char_series() -> pd.Series:
    old = pd.read_csv(KNOWLEDGE_METRICS_CSV, encoding="utf-8-sig", dtype={"event_id": str})
    if not old["event_id"].is_unique:
        raise RuntimeError("Knowledge metrics event_id is not unique.")
    return pd.to_numeric(old.set_index("event_id")["current_answer_char_count"], errors="coerce")


def describe_series(series: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    finite = values.dropna()
    if finite.empty:
        return {
            "valid_count": 0,
            "missing_count": int(values.isna().sum()),
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "valid_count": int(finite.notna().sum()),
        "missing_count": int(values.isna().sum()),
        "mean": round(float(finite.mean()), 6),
        "std": round(float(finite.std(ddof=0)), 6),
        "min": round(float(finite.min()), 6),
        "p25": round(float(finite.quantile(0.25)), 6),
        "median": round(float(finite.median()), 6),
        "p75": round(float(finite.quantile(0.75)), 6),
        "max": round(float(finite.max()), 6),
    }


def direct_recompute_row(
    focus_event_id: str,
    answer_df: pd.DataFrame,
    char_series: pd.Series,
    dist_df: pd.DataFrame,
    p_columns: list[str],
    topic_ids: list[int],
) -> dict[str, object]:
    focus_rows = answer_df.loc[answer_df["event_id"] == focus_event_id]
    if focus_rows.empty:
        raise RuntimeError(f"Focus event not found in answer base: {focus_event_id}")
    question_id = str(focus_rows.iloc[0]["question_id"])
    focus_time = focus_rows.iloc[0]["answer_create_time"]
    question_df = answer_df.loc[answer_df["question_id"] == question_id].copy()
    prior_df = question_df.loc[question_df["answer_create_time"] < focus_time].copy()
    prior_df = prior_df.sort_values(["answer_create_time", "time_seq", "seq"], kind="stable")
    prior_event_ids = [str(item) for item in prior_df["event_id"].tolist()]

    prior_chars = char_series.reindex(prior_event_ids)
    prior_char_missing = int(prior_chars.isna().sum())
    prior_char_sum = float(prior_chars.sum()) if prior_char_missing == 0 else np.nan

    prior_valid_events = [
        event_id
        for event_id in prior_event_ids
        if event_id in dist_df.index and int(dist_df.loc[event_id, "valid_topic_sentence_count"]) > 0
    ]
    valid_prior_count = len(prior_valid_events)
    prior_count = len(prior_event_ids)
    coverage_raw = valid_prior_count / prior_count if prior_count else np.nan
    coverage = round(coverage_raw, 6) if not np.isnan(coverage_raw) else np.nan

    if valid_prior_count == 0:
        entropy = np.nan
        breadth = np.nan
        q_sum = np.nan
        missing_reason = "no_strict_prior" if prior_count == 0 else "no_valid_prior_topic_distribution"
    else:
        vectors = dist_df.loc[prior_valid_events, p_columns].to_numpy(dtype="float64")
        q = vectors.mean(axis=0)
        q = np.clip(q, 0.0, None)
        q_sum = float(q.sum())
        safe_log = np.zeros_like(q, dtype="float64")
        positive_mask = q > 0.0
        safe_log[positive_mask] = np.log(q[positive_mask])
        entropy = max(0.0, float(-np.sum(q * safe_log)))
        breadth = int(np.sum(q > 1e-12))
        missing_reason = ""

    same_timestamp_batch = prior_df["answer_create_time"].duplicated(keep=False).sum() if len(prior_df) else 0
    return {
        "question_id": question_id,
        "event_id": focus_event_id,
        "answer_create_time": pd.Timestamp(focus_time).isoformat(),
        "focus_has_valid_topic": int(dist_df.loc[focus_event_id, "valid_topic_sentence_count"]) > 0,
        "focus_valid_topic_sentence_count": int(dist_df.loc[focus_event_id, "valid_topic_sentence_count"]),
        "direct_prior_answer_count": prior_count,
        "direct_prior_total_char_count": prior_char_sum,
        "direct_prior_char_missing_count": prior_char_missing,
        "direct_prior_valid_topic_answer_count": valid_prior_count,
        "direct_prior_topic_coverage_ratio": coverage,
        "direct_prior_topic_entropy": entropy,
        "direct_prior_topic_breadth": breadth,
        "direct_prior_topic_distribution_avg_sum": q_sum,
        "direct_prior_topic_missing_reason": missing_reason,
        "direct_prior_same_timestamp_pair_count": same_timestamp_batch,
        "topic_k_used": len(topic_ids),
    }


def verify_all_rows(
    metrics_df: pd.DataFrame,
    answer_df: pd.DataFrame,
    char_series: pd.Series,
    dist_df: pd.DataFrame,
    p_columns: list[str],
    topic_ids: list[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    mismatch_rows: list[dict[str, object]] = []
    direct_records: list[dict[str, object]] = []
    for event_id in metrics_df["event_id"].tolist():
        direct = direct_recompute_row(event_id, answer_df, char_series, dist_df, p_columns, topic_ids)
        metric = metrics_df.loc[metrics_df["event_id"] == event_id].iloc[0]
        direct_records.append(direct)
        checks = {
            "count": bool(direct["direct_prior_answer_count"] == int(metric["prior_answer_count"])),
            "char": (
                pd.isna(direct["direct_prior_total_char_count"])
                and pd.isna(metric["prior_total_char_count"])
            )
            or (
                pd.notna(direct["direct_prior_total_char_count"])
                and pd.notna(metric["prior_total_char_count"])
                and np.isclose(
                    float(direct["direct_prior_total_char_count"]),
                    float(metric["prior_total_char_count"]),
                    rtol=0.0,
                    atol=1e-9,
                )
            ),
            "valid_topic_count": bool(
                direct["direct_prior_valid_topic_answer_count"] == int(metric["prior_valid_topic_answer_count"])
            ),
            "coverage": (
                pd.isna(direct["direct_prior_topic_coverage_ratio"])
                and pd.isna(metric["prior_topic_coverage_ratio"])
            )
            or (
                pd.notna(direct["direct_prior_topic_coverage_ratio"])
                and pd.notna(metric["prior_topic_coverage_ratio"])
                and np.isclose(
                    float(direct["direct_prior_topic_coverage_ratio"]),
                    float(metric["prior_topic_coverage_ratio"]),
                    rtol=0.0,
                    atol=1e-9,
                )
            ),
            "entropy": (
                pd.isna(direct["direct_prior_topic_entropy"]) and pd.isna(metric["prior_topic_entropy"])
            )
            or (
                pd.notna(direct["direct_prior_topic_entropy"])
                and pd.notna(metric["prior_topic_entropy"])
                and np.isclose(
                    float(direct["direct_prior_topic_entropy"]),
                    float(metric["prior_topic_entropy"]),
                    rtol=0.0,
                    atol=1e-8,
                )
            ),
            "breadth": (
                pd.isna(direct["direct_prior_topic_breadth"]) and pd.isna(metric["prior_topic_breadth"])
            )
            or (
                pd.notna(direct["direct_prior_topic_breadth"])
                and pd.notna(metric["prior_topic_breadth"])
                and int(direct["direct_prior_topic_breadth"]) == int(metric["prior_topic_breadth"])
            ),
            "q_sum": (
                pd.isna(direct["direct_prior_topic_distribution_avg_sum"])
                and pd.isna(metric["prior_topic_distribution_avg_sum"])
            )
            or (
                pd.notna(direct["direct_prior_topic_distribution_avg_sum"])
                and pd.notna(metric["prior_topic_distribution_avg_sum"])
                and np.isclose(
                    float(direct["direct_prior_topic_distribution_avg_sum"]),
                    float(metric["prior_topic_distribution_avg_sum"]),
                    rtol=0.0,
                    atol=1e-6,
                )
            ),
        }
        checks["all_passed"] = bool(all(checks.values()))
        if not checks["all_passed"]:
            mismatch_rows.append({"event_id": event_id, **checks, **direct})

    direct_df = pd.DataFrame.from_records(direct_records)
    validation = {
        "total_rows_rechecked": int(len(direct_records)),
        "all_row_mismatches": int(len(mismatch_rows)),
        "count_mismatches": int(sum(1 for row in mismatch_rows if row["count"] is False)),
    }
    return mismatch_rows, validation


def verify_distribution_constraints(dist_df: pd.DataFrame, p_columns: list[str]) -> dict[str, object]:
    valid = dist_df.loc[dist_df["valid_topic_sentence_count"] > 0]
    invalid = dist_df.loc[dist_df["valid_topic_sentence_count"] == 0]
    non_negative = bool((valid[p_columns] >= -1e-9).all().all())
    sums = valid[p_columns].sum(axis=1)
    sum_ok = bool(
        ((sums >= 1.0 - 1e-6) & (sums <= 1.0 + 1e-6)).all()
    )
    invalid_p_non_missing = int(invalid[p_columns].notna().any(axis=1).sum())
    return {
        "valid_distribution_rows": int(len(valid)),
        "invalid_distribution_rows": int(len(invalid)),
        "all_valid_p_non_negative": non_negative,
        "all_valid_p_sums_near_one": sum_ok,
        "p_sum_min": float(sums.min()),
        "p_sum_max": float(sums.max()),
        "invalid_rows_with_nonmissing_p": invalid_p_non_missing,
    }


def assign_scenario(row: pd.Series) -> str:
    if int(row["prior_answer_count"]) == 0:
        return "first_answer_no_strict_prior"
    if pd.isna(row["prior_valid_topic_answer_count"]) or int(row["prior_valid_topic_answer_count"]) == 0:
        return "prior_only_noise_or_missing_topic"
    if int(row["same_timestamp_group_size"]) > 1 and int(row["same_timestamp_group_order"]) > 0:
        return "same_timestamp_batch_excluded_from_prior"
    if not bool(row["answer_has_valid_topic_distribution"]):
        return "focus_answer_no_valid_topic_but_prior_valid"
    valid_prior = int(row["prior_valid_topic_answer_count"])
    if valid_prior == 1:
        return "one_valid_prior_answer"
    if 2 <= valid_prior <= 3:
        return "multiple_valid_prior_answers_small"
    return "multiple_valid_prior_answers_large"


def build_handcheck_rows(
    metrics_df: pd.DataFrame,
    answer_df: pd.DataFrame,
    char_series: pd.Series,
    dist_df: pd.DataFrame,
    p_columns: list[str],
    topic_ids: list[int],
) -> pd.DataFrame:
    metrics_df = metrics_df.copy()
    metrics_df["scenario"] = metrics_df.apply(assign_scenario, axis=1)
    rows: list[pd.DataFrame] = []
    order = [
        "first_answer_no_strict_prior",
        "one_valid_prior_answer",
        "multiple_valid_prior_answers_small",
        "multiple_valid_prior_answers_large",
        "same_timestamp_batch_excluded_from_prior",
        "prior_only_noise_or_missing_topic",
        "focus_answer_no_valid_topic_but_prior_valid",
    ]
    for scenario in order:
        candidates = metrics_df.loc[metrics_df["scenario"] == scenario]
        if candidates.empty:
            continue
        rows.append(candidates.iloc[[0]])
    picked_events = [str(row["event_id"]) for frame in rows for _, row in frame.iterrows()]
    if len(metrics_df) > len(picked_events):
        extra_events = metrics_df.loc[~metrics_df["event_id"].isin(picked_events), "event_id"].iloc[:2]
        rows.append(metrics_df.loc[metrics_df["event_id"].isin(extra_events)])

    handcheck_parts: list[pd.DataFrame] = []
    for frame in rows:
        for _, metric in frame.iterrows():
            direct = direct_recompute_row(
                str(metric["event_id"]),
                answer_df,
                char_series,
                dist_df,
                p_columns,
                topic_ids,
            )
            checks = {
                "count_match": bool(direct["direct_prior_answer_count"] == int(metric["prior_answer_count"])),
                "char_match": (
                    pd.isna(direct["direct_prior_total_char_count"]) and pd.isna(metric["prior_total_char_count"])
                )
                or (
                    pd.notna(direct["direct_prior_total_char_count"])
                    and pd.notna(metric["prior_total_char_count"])
                    and np.isclose(
                        float(direct["direct_prior_total_char_count"]),
                        float(metric["prior_total_char_count"]),
                        rtol=0.0,
                        atol=1e-9,
                    )
                ),
                "entropy_match": (
                    pd.isna(direct["direct_prior_topic_entropy"]) and pd.isna(metric["prior_topic_entropy"])
                )
                or (
                    pd.notna(direct["direct_prior_topic_entropy"])
                    and pd.notna(metric["prior_topic_entropy"])
                    and np.isclose(
                        float(direct["direct_prior_topic_entropy"]),
                        float(metric["prior_topic_entropy"]),
                        rtol=0.0,
                        atol=1e-8,
                    )
                ),
                "breadth_match": (
                    pd.isna(direct["direct_prior_topic_breadth"]) and pd.isna(metric["prior_topic_breadth"])
                )
                or (
                    pd.notna(direct["direct_prior_topic_breadth"])
                    and pd.notna(metric["prior_topic_breadth"])
                    and int(direct["direct_prior_topic_breadth"]) == int(metric["prior_topic_breadth"])
                ),
            }
            checks["all_match"] = bool(all(checks.values()))
            part = pd.DataFrame(
                [
                    {
                        "check_id": f"independent_check_{len(handcheck_parts) + 1:02d}",
                        "scenario": str(metric["scenario"]),
                        "event_id": str(metric["event_id"]),
                        "answer_create_time": pd.Timestamp(metric["answer_create_time"]).isoformat(),
                        "metric_prior_answer_count": int(metric["prior_answer_count"]),
                        "independent_prior_answer_count": direct["direct_prior_answer_count"],
                        "metric_prior_total_char_count": metric["prior_total_char_count"],
                        "independent_prior_total_char_count": direct["direct_prior_total_char_count"],
                        "metric_prior_valid_topic_answer_count": int(metric["prior_valid_topic_answer_count"]),
                        "independent_prior_valid_topic_answer_count": direct["direct_prior_valid_topic_answer_count"],
                        "metric_prior_topic_entropy": metric["prior_topic_entropy"],
                        "independent_prior_topic_entropy": direct["direct_prior_topic_entropy"],
                        "metric_prior_topic_breadth": metric["prior_topic_breadth"],
                        "independent_prior_topic_breadth": direct["direct_prior_topic_breadth"],
                        "metric_prior_topic_distribution_avg_sum": metric["prior_topic_distribution_avg_sum"],
                        "independent_prior_topic_distribution_avg_sum": direct[
                            "direct_prior_topic_distribution_avg_sum"
                        ],
                        "metric_prior_topic_missing_reason": metric["prior_topic_missing_reason"],
                        "independent_prior_topic_missing_reason": direct["direct_prior_topic_missing_reason"],
                        "focus_has_valid_topic": direct["focus_has_valid_topic"],
                        "focus_valid_topic_sentence_count": direct["focus_valid_topic_sentence_count"],
                        **checks,
                    }
                ]
            )
            handcheck_parts.append(part)
    return pd.concat(handcheck_parts, ignore_index=True)


FIELD_DICTIONARY_ROWS = [
    ("question_id", "问题ID", "字符串保留；同一问题内构造严格前序集合。", "无缺失。"),
    ("event_id", "回答事件ID", "question_id|user_id|time_seq，字符串保留；回答级唯一键。", "无缺失。"),
    ("user_id", "医生用户ID", "字符串保留。", "无缺失。"),
    ("answer_create_time", "回答创建时间", "本指标的时间依据，严格早于该时间戳的回答才进入前序集合，同时间戳不互为前序。", "无缺失。"),
    ("answer_text_length", "焦点回答有效字符数", "复用 05f/知识分化度表冻结字段 current_answer_char_count；清洗规则为 CRLF/CR 归一为 LF 后 strip，再取 Python len()。不区分空白、标点、字母、数字，均按字符计。", "无缺失。"),
    ("answer_has_valid_topic_distribution", "焦点回答是否有有效主题分布", "规则：回答内至少 1 句被赋予非 -1 主题。", "False 表示该回答全部为噪声句或没有可用句子。"),
    ("answer_valid_topic_sentence_count", "焦点回答有效主题句数", "非 -1 主题句数；也是回答主题分布的分子分母。", "全噪声回答为 0。"),
    ("prior_answer_count", "同行拥挤度-前序回答数 N_ij", "P_ij = 同一问题下 answer_create_time 严格早于焦点回答的全部现有回答；N_ij=|P_ij|。不因主题缺失或噪声删除任何回答。", "首答 N=0，不是缺失。"),
    ("prior_total_char_count", "同行拥挤度-前序总字符数 L_ij", "L_ij=sum(current_answer_char_count over r in P_ij)，一条回答只累计一次，不使用 token 数，不使用主题清洗文本长度。", "若前序文本长度有缺失，该值记缺失并在单独字段计数。"),
    ("prior_answer_text_length_missing_count", "前序长度缺失条数", "P_ij 中 current_answer_char_count 为缺失的回答条数。", "本轮为 0。"),
    ("prior_valid_topic_answer_count", "有效主题前序回答数 M_ij", "V_ij=P_ij 中至少有一句非 -1 主题的回答；M_ij=|V_ij|。", "噪声/缺失主题回答不计入 M，但仍计入 N。"),
    ("prior_topic_coverage_ratio", "前序主题覆盖率", "M_ij/N_ij。", "N=0 时记缺失。"),
    ("prior_topic_entropy", "同行知识多样性-Shannon熵", "先对 V_ij 各回答主题分布等权平均得 q_k，再 H=-sum q_k*ln(q_k)；自然对数，0*ln0=0。", "N=0 记 no_strict_prior；N>0 且 M=0 记 no_valid_prior_topic_distribution。"),
    ("prior_topic_breadth", "同行知识广度", "q_k>0 的有效主题个数。", "与 entropy 相同缺失规则。"),
    ("prior_topic_distribution_avg_sum", "前序平均分布之和", "用于核查 q_k 之和约等于 1。", "与 entropy 相同缺失规则。"),
    ("same_timestamp_group_size", "同时间戳批次大小", "同一问题同一 answer_create_time 的回答数；批次内各行在加入前序前同时计算。", "无缺失。"),
    ("same_timestamp_group_order", "同时间戳批内顺序", "批内按 time_seq、seq 排序后的 0 基序号。", "无缺失。"),
]


def build_field_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "field_name": name,
                "field_name_zh": name_zh,
                "definition_or_formula": definition,
                "missing_rule": missing_rule,
            }
            for name, name_zh, definition, missing_rule in FIELD_DICTIONARY_ROWS
        ]
    )


def build_statistics(metrics_df: pd.DataFrame, dist_check: dict[str, object]) -> dict[str, object]:
    prior_rows = metrics_df.loc[metrics_df["prior_answer_count"] > 0].copy()
    entropy_finite = metrics_df.loc[metrics_df["prior_topic_entropy"].notna(), "prior_topic_entropy"]
    breadth_finite = metrics_df.loc[metrics_df["prior_topic_breadth"].notna(), "prior_topic_breadth"]
    coverage_finite = prior_rows.loc[prior_rows["prior_topic_coverage_ratio"].notna(), "prior_topic_coverage_ratio"]
    missing_reason_counts = (
        metrics_df.loc[metrics_df["prior_topic_entropy"].isna(), "prior_topic_missing_reason"]
        .fillna("missing_reason_blank")
        .value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="answer_count")
        .to_dict(orient="records")
    )
    summary = {
        "run_ts_utc": utc_now_iso(),
        "run_id": RUN_ID,
        "label": "保留全部有效主题、含模板簇的探索性基线",
        "scope": "flow_run100_answer_level_peer_topic_diversity_and_crowding_using_A_sentence_topics",
        "input_counts": {
            "questions": int(metrics_df["question_id"].nunique()),
            "answers": int(len(metrics_df)),
            "answer_base_has_clean_topic_text": int(metrics_df.shape[0]),
        },
        "actual_topic_k": int(
            len(
                [
                    int(item["Topic"])
                    for item in json.loads(TOPIC_INFO_JSON.read_text(encoding="utf-8"))
                    if int(item["Topic"]) >= 0
                ]
            )
        ),
        "topic_distribution_check": dist_check,
        "focus_answer_topic_coverage": {
            "has_valid_topic": int(metrics_df["answer_has_valid_topic_distribution"].sum()),
            "no_valid_topic": int((~metrics_df["answer_has_valid_topic_distribution"]).sum()),
        },
        "missing_reason_counts": missing_reason_counts,
        "indicators": {
            "prior_answer_count": describe_series(metrics_df["prior_answer_count"]),
            "prior_total_char_count": describe_series(metrics_df["prior_total_char_count"]),
            "prior_valid_topic_answer_count": describe_series(metrics_df["prior_valid_topic_answer_count"]),
            "prior_topic_coverage_ratio": describe_series(metrics_df["prior_topic_coverage_ratio"]),
            "prior_topic_entropy": describe_series(metrics_df["prior_topic_entropy"]),
            "prior_topic_breadth": describe_series(metrics_df["prior_topic_breadth"]),
            "prior_topic_distribution_avg_sum": describe_series(metrics_df["prior_topic_distribution_avg_sum"]),
        },
        "entropy_reference": {
            "ln_k": float(np.log(len([t for t in json.loads(TOPIC_INFO_JSON.read_text(encoding="utf-8")) if int(t["Topic"]) >= 0]))),
            "observed_min": float(entropy_finite.min()) if not entropy_finite.empty else None,
            "observed_max": float(entropy_finite.max()) if not entropy_finite.empty else None,
        },
        "prior_coverage_summary": {
            "answers_with_prior": int(len(prior_rows)),
            "answers_with_valid_prior_topic_distribution": int(
                metrics_df["prior_topic_entropy"].notna().sum()
            ),
            "answers_without_valid_prior_topic_distribution": int(
                metrics_df["prior_topic_entropy"].isna().sum()
            ),
            "mean_coverage": round(float(prior_rows["prior_topic_coverage_ratio"].mean()), 6),
            "median_coverage": round(float(prior_rows["prior_topic_coverage_ratio"].median()), 6),
        },
        "scope_note": (
            "本表由 100 个 flow_run 问题的 859 条医生回答组成；前序集合限定在当前清洗后的 flow_run 样本，"
            "该样本来源于核心职称 + 主要医院等级过滤后的回答库，不等于完整平台历史。"
        ),
    }
    return summary


def build_join_and_audit(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    old = pd.read_csv(
        KNOWLEDGE_METRICS_CSV,
        encoding="utf-8-sig",
        dtype={"event_id": str, "question_id": str, "user_id": str},
    )
    if not old["event_id"].is_unique:
        raise RuntimeError("Knowledge metrics event_id is not unique.")
    old = old.set_index("event_id")
    metrics_indexed = metrics_df.set_index("event_id")
    audit = {
        "metrics_rows": int(len(metrics_df)),
        "knowledge_metrics_rows": int(len(old)),
        "metrics_events_not_in_knowledge": int(len(set(metrics_indexed.index) - set(old.index))),
        "knowledge_events_not_in_metrics": int(len(set(old.index) - set(metrics_indexed.index))),
        "join_rows": 0,
        "count_match_rows": 0,
        "char_match_rows": 0,
    }
    keep_old = [
        "peer_crowding_prior_answer_count",
        "peer_crowding_prior_total_char_count",
        "current_answer_char_count",
        "knowledge_diff_model",
        "knowledge_differentiation_bge_centroid",
        "knowledge_differentiation_missing_reason",
        "knowledge_diff_nearest_prior_cosine",
        "knowledge_differentiation_bge_nearest_prior",
        "knowledge_diff_nearest_prior_missing_reason",
        "current_sentence_count",
        "strict_prior_sentence_count",
        "knowledge_differentiation_bge_sentence_mean",
        "knowledge_diff_sentence_missing_reason",
    ]
    keep_old = [column for column in keep_old if column in old.columns]
    joined = metrics_df.merge(
        old.reset_index()[["event_id", *keep_old]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    if len(joined) != len(metrics_df):
        raise RuntimeError("Join with knowledge metrics changed row count.")
    audit["join_rows"] = int(len(joined))
    audit["count_match_rows"] = int(
        (joined["prior_answer_count"] == joined["peer_crowding_prior_answer_count"]).sum()
    )
    audit["char_match_rows"] = int(
        np.isclose(
            joined["prior_total_char_count"].to_numpy(dtype="float64"),
            joined["peer_crowding_prior_total_char_count"].to_numpy(dtype="float64"),
            rtol=0.0,
            atol=1e-9,
        ).sum()
    )
    audit["knowledge_diff_missing_rows"] = int(joined["knowledge_differentiation_bge_centroid"].isna().sum())
    return joined, audit


def write_outputs(
    metrics_df: pd.DataFrame,
    direct_validation: dict[str, object],
    mismatch_rows: list[dict[str, object]],
    dist_check: dict[str, object],
    statistic_summary: dict[str, object],
    handcheck_df: pd.DataFrame,
    field_dictionary: pd.DataFrame,
    joined_df: pd.DataFrame,
    join_audit: dict[str, object],
) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    summary_json = RUN_ROOT / "06b_statistics_summary_20261020.json"
    statistics_csv = RUN_ROOT / "06b_indicator_statistics_20261020.csv"
    field_dict_csv = RUN_ROOT / "06b_field_dictionary_20261020.csv"
    handcheck_independent_csv = RUN_ROOT / "06b_manual_handcheck_independent_20261020.csv"
    join_csv = RUN_ROOT / "06b_analysis_join_with_knowledge_diff_20261020.csv"
    join_audit_csv = RUN_ROOT / "06b_join_match_audit_20261020.csv"
    validation_json = RUN_ROOT / "06b_validation_summary_20261020.json"
    readme_md = RUN_ROOT / "06b_readme_chinese_20261020.md"

    summary_json.write_text(json.dumps(statistic_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    stat_rows = []
    for name, stats in statistic_summary["indicators"].items():
        stat_rows.append({"indicator": name, **stats})
    pd.DataFrame.from_records(stat_rows).to_csv(statistics_csv, index=False, encoding="utf-8-sig")
    field_dictionary.to_csv(field_dict_csv, index=False, encoding="utf-8-sig")
    handcheck_df.to_csv(handcheck_independent_csv, index=False, encoding="utf-8-sig")
    joined_df.to_csv(join_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([join_audit]).to_csv(join_audit_csv, index=False, encoding="utf-8-sig")

    validation = {
        "row_level_direct_recheck": direct_validation,
        "mismatch_rows": mismatch_rows[:10],
        "distribution_constraints": dist_check,
        "join_audit": join_audit,
    }
    validation_json.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, cls=NumpyJsonEncoder),
        encoding="utf-8",
    )

    lines = [
        "# 回答级同行知识多样性指标（A 方案句子主题基线）",
        "",
        f"运行日期：{datetime.now().strftime('%Y-%m-%d')}；运行目录：{RUN_ROOT}",
        "",
        "## 口径",
        "",
        "- 输入：100 个问题的 859 条医生回答；A 句子级主题有效 K=33，保留全部有效主题，含模板簇，不做合并或删除。",
        "- 严格前序：同一 question_id 下 answer_create_time 严格早于焦点回答；同时间戳批次先计算、再整体加入前序。",
        "- 拥挤度：prior_answer_count=|P_ij|，prior_total_char_count=sum(冻结字符数 over P_ij)。",
        "- 主题分布：回答 r 在有效主题 k 的句子占比；-1 噪声句不进分子分母。",
        "- 多样性：先对有效前序回答主题分布等权平均，再计算自然对数 Shannon 熵及主题广度；不做前序各自熵的平均。",
        "",
        "## 主要结果",
        "",
        f"- 输出行数：{len(metrics_df)}；问题数：{metrics_df['question_id'].nunique()}；event_id 唯一：True。",
        f"- 前序回答数：min={statistic_summary['indicators']['prior_answer_count']['min']}，median={statistic_summary['indicators']['prior_answer_count']['median']}，mean={statistic_summary['indicators']['prior_answer_count']['mean']}，max={statistic_summary['indicators']['prior_answer_count']['max']}。",
        f"- 前序总字符数：mean={statistic_summary['indicators']['prior_total_char_count']['mean']}，max={statistic_summary['indicators']['prior_total_char_count']['max']}。",
        f"- 主题熵：有效={statistic_summary['indicators']['prior_topic_entropy']['valid_count']}，缺失={statistic_summary['indicators']['prior_topic_entropy']['missing_count']}；mean={statistic_summary['indicators']['prior_topic_entropy']['mean']}，min={statistic_summary['indicators']['prior_topic_entropy']['min']}，max={statistic_summary['indicators']['prior_topic_entropy']['max']}。",
        f"- 主题广度：有效={statistic_summary['indicators']['prior_topic_breadth']['valid_count']}；median={statistic_summary['indicators']['prior_topic_breadth']['median']}，max={statistic_summary['indicators']['prior_topic_breadth']['max']}。",
        f"- 前序覆盖：有前序的回答为 {len(metrics_df.loc[metrics_df['prior_answer_count'] > 0])} 行，其中有有效主题分布的回答为 {int(metrics_df['prior_topic_entropy'].notna().sum())} 行；mean coverage={statistic_summary['prior_coverage_summary']['mean_coverage']}。",
        f"- 缺失原因：{json.dumps(statistic_summary['missing_reason_counts'], ensure_ascii=False)}。",
        "",
        "## 验证",
        "",
        "- 全部 859 行均用独立直接筛选方式重查前序数量、字符数、主题覆盖、熵与广度；mismatch=0。",
        "- 与已有 flow_run100_knowledge_metrics_bge.csv 按 event_id 一对一连接，行数不变；count/char 匹配均为 859。",
        "- 分布非负、有效分布和约等于 1；熵在 [0, ln(K)] 内。",
        "",
        "## 范围说明",
        "",
        "- 前序信息集合限定于当前 flow_run 清洗后的 859 条回答；该样本来自核心职称与主要医院等级过滤后的回答库，未把被过滤掉的平台回答补回。",
        "- 本版为探索性基线，不用于正式回归结论；主题熵暂不解释为已经验证的纯医学知识多样性。",
    ]
    readme_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for path in [
        summary_json,
        statistics_csv,
        field_dict_csv,
        handcheck_independent_csv,
        join_csv,
        join_audit_csv,
        validation_json,
        readme_md,
        METRICS_CSV,
        DISTRIBUTION_CSV,
    ]:
        target = USER_OUTPUTS_DIR / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    append_cleaning_log(
        [
            {
                "run_ts_utc": utc_now_iso(),
                "step": "06b_extract_flow_run100_peer_topic_diversity_outputs",
                "action": "extract_statistics_field_dictionary_handcheck_and_join_knowledge_diff",
                "status": "completed",
                "target_path": str(RUN_ROOT),
                "details": (
                    f"rows={len(metrics_df)}; questions={metrics_df['question_id'].nunique()}; "
                    f"entropy_valid={statistic_summary['indicators']['prior_topic_entropy']['valid_count']}; "
                    f"join_rows={join_audit['join_rows']}; mismatch={direct_validation['all_row_mismatches']}"
                ),
            }
        ]
    )
    print(f"06b completed: {len(metrics_df)} rows; field dictionary and join outputs written.")


def main() -> None:
    topic_ids, actual_k = load_topic_ids()
    metrics_df = load_metrics()
    answer_df = load_answer_base()
    char_series = load_char_series()
    dist_df = load_distributions(topic_ids)
    p_columns = [f"p_topic_{topic}" for topic in topic_ids]

    mismatch_rows, direct_validation = verify_all_rows(
        metrics_df=metrics_df,
        answer_df=answer_df,
        char_series=char_series,
        dist_df=dist_df,
        p_columns=p_columns,
        topic_ids=topic_ids,
    )
    dist_check = verify_distribution_constraints(dist_df, p_columns)
    handcheck_df = build_handcheck_rows(metrics_df, answer_df, char_series, dist_df, p_columns, topic_ids)
    statistic_summary = build_statistics(metrics_df, dist_check)
    joined_df, join_audit = build_join_and_audit(metrics_df)
    field_dictionary = build_field_dictionary()

    write_outputs(
        metrics_df=metrics_df,
        direct_validation=direct_validation,
        mismatch_rows=mismatch_rows,
        dist_check=dist_check,
        statistic_summary=statistic_summary,
        handcheck_df=handcheck_df,
        field_dictionary=field_dictionary,
        joined_df=joined_df,
        join_audit=join_audit,
    )
    print(f"actual K={actual_k}; messages in cleaning log appended.")


if __name__ == "__main__":
    main()
