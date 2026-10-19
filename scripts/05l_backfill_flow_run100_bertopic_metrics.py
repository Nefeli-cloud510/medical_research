from __future__ import annotations

import csv
import importlib.util
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
TABLE_DIR = ROOT_DIR / "outputs" / "tables"
METADATA_DIR = ROOT_DIR / "metadata"

KNOWLEDGE_PATH = ANALYSIS_DIR / "flow_run100_knowledge_metrics_bge.parquet"
ANSWER_PATH = ANALYSIS_DIR / "flow_run100_answer_text_features.parquet"
SUMMARY_PATH = AUDIT_DIR / "20g_flow_run100_bertopic_metrics_summary.json"
TOPIC_INFO_PATH = TABLE_DIR / "20g_flow_run100_bertopic_topic_info.csv"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"
LOCAL_MODEL_DIR = ROOT_DIR / "models" / "bge-base-zh-v1.5"

MODEL_NAME = "BAAI/bge-base-zh-v1.5"
SEED = 20261007
BATCH_SIZE = 64
OUTLIER_TOPIC_ID = -1
MISSING_REASON_NO_PRIOR = "no_strict_prior"
MISSING_REASON_NO_VALID_PRIOR = "no_valid_prior_topics"
MISSING_REASON_CURRENT_OUTLIER = "current_topic_outlier"

TOPIC_OUTPUT_COLUMNS = [
    "bertopic_topic_id_answer",
    "bertopic_topic_name_answer",
    "bertopic_topic_probability_answer",
    "topic_prior_valid_answer_count_bertopic",
    "topic_prior_unique_topic_count_bertopic",
    "topic_prior_outlier_answer_count_bertopic",
    "topic_prior_outlier_ratio_bertopic",
    "topic_prior_entropy_bertopic",
    "topic_prior_entropy_bertopic_norm",
    "topic_knowledge_diff_bertopic_jsd",
    "topic_knowledge_diff_bertopic_jsd_norm",
    "topic_knowledge_diff_bertopic_current_topic_prior_share",
    "topic_knowledge_diff_bertopic_missing_reason",
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


def choose_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_embedding_model() -> tuple[SentenceTransformer, str, str]:
    device = choose_device()
    model_source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_NAME
    model = SentenceTransformer(model_source, device=device)
    return model, device, model_source


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32, copy=False)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    knowledge_df = pd.read_parquet(KNOWLEDGE_PATH).copy()
    answer_df = pd.read_parquet(ANSWER_PATH).copy()

    knowledge_df["question_id"] = knowledge_df["question_id"].astype(str)
    answer_df["question_id"] = answer_df["question_id"].astype(str)
    knowledge_df["answer_create_time"] = pd.to_datetime(knowledge_df["answer_create_time"])
    answer_df["answer_create_time"] = pd.to_datetime(answer_df["answer_create_time"])

    answer_df = answer_df.sort_values(
        ["question_id", "answer_create_time", "time_seq", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    return knowledge_df, answer_df


def build_topic_model():
    original_find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name, *args, **kwargs: (  # type: ignore[assignment]
        None if name == "plotly" else original_find_spec(name, *args, **kwargs)
    )
    try:
        from bertopic import BERTopic
    finally:
        importlib.util.find_spec = original_find_spec

    vectorizer_model = CountVectorizer(
        tokenizer=jieba.lcut,
        token_pattern=None,
        lowercase=False,
        min_df=2,
    )
    umap_model = TruncatedSVD(
        n_components=5,
        random_state=SEED,
    )
    hdbscan_model = KMeans(
        n_clusters=12,
        random_state=SEED,
        n_init=20,
    )
    topic_model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )
    return topic_model


def safe_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return float("nan")
    return float(-(positive * np.log(positive)).sum())


def js_divergence_from_onehot(prior_probabilities: np.ndarray, current_index: int) -> float:
    current = np.zeros_like(prior_probabilities, dtype=np.float64)
    current[current_index] = 1.0
    midpoint = 0.5 * (current + prior_probabilities)
    return safe_entropy(midpoint) - 0.5 * safe_entropy(prior_probabilities)


def topic_name_lookup_df(topic_model) -> pd.DataFrame:
    topic_info_df = topic_model.get_topic_info().copy()
    topic_info_df["Topic"] = topic_info_df["Topic"].astype(int)
    return topic_info_df


def compute_topic_metrics(
    answer_df: pd.DataFrame,
    topic_ids: list[int],
    topic_probabilities: np.ndarray | None,
    valid_topic_ids: list[int],
    topic_name_lookup: dict[int, str],
) -> pd.DataFrame:
    if not valid_topic_ids:
        raise RuntimeError("BERTopic 未产生任何有效主题，无法计算前序主题熵。")

    global_topic_to_index = {topic_id: idx for idx, topic_id in enumerate(valid_topic_ids)}
    max_jsd = math.log(2.0)
    max_entropy = math.log(len(valid_topic_ids)) if len(valid_topic_ids) > 1 else float("nan")

    work_df = answer_df.copy()
    work_df["bertopic_topic_id_answer"] = topic_ids
    if topic_probabilities is None:
        work_df["bertopic_topic_probability_answer"] = np.nan
    else:
        work_df["bertopic_topic_probability_answer"] = topic_probabilities.max(axis=1).astype(float)
    work_df["bertopic_topic_name_answer"] = work_df["bertopic_topic_id_answer"].map(
        lambda x: topic_name_lookup.get(int(x), "")
    )

    rows: list[dict[str, object]] = []

    for question_id, group_df in work_df.groupby("question_id", sort=False):
        group_df = group_df.sort_values(
            ["answer_create_time", "time_seq", "seq"],
            kind="stable",
        )

        prior_total_count = 0
        prior_outlier_count = 0
        prior_valid_counter: Counter[int] = Counter()

        for _, timestamp_df in group_df.groupby("answer_create_time", sort=False):
            timestamp_df = timestamp_df.sort_values(["time_seq", "seq"], kind="stable")

            for row in timestamp_df.itertuples(index=False):
                prior_valid_count = int(sum(prior_valid_counter.values()))
                prior_unique_topic_count = int(len(prior_valid_counter))
                prior_outlier_ratio = (
                    float(prior_outlier_count / prior_total_count) if prior_total_count > 0 else np.nan
                )

                if prior_total_count == 0:
                    prior_entropy = np.nan
                    prior_entropy_norm = np.nan
                    topic_jsd = np.nan
                    topic_jsd_norm = np.nan
                    current_topic_prior_share = np.nan
                    missing_reason = MISSING_REASON_NO_PRIOR
                elif prior_valid_count == 0:
                    prior_entropy = np.nan
                    prior_entropy_norm = np.nan
                    topic_jsd = np.nan
                    topic_jsd_norm = np.nan
                    current_topic_prior_share = np.nan
                    missing_reason = MISSING_REASON_NO_VALID_PRIOR
                else:
                    prior_probability_vector = np.zeros(len(valid_topic_ids), dtype=np.float64)
                    for topic_id, count in prior_valid_counter.items():
                        prior_probability_vector[global_topic_to_index[topic_id]] = count / prior_valid_count

                    prior_entropy = safe_entropy(prior_probability_vector)
                    if math.isnan(max_entropy) or max_entropy <= 0.0:
                        prior_entropy_norm = np.nan
                    else:
                        prior_entropy_norm = float(prior_entropy / max_entropy)

                    if int(row.bertopic_topic_id_answer) == OUTLIER_TOPIC_ID:
                        topic_jsd = np.nan
                        topic_jsd_norm = np.nan
                        current_topic_prior_share = np.nan
                        missing_reason = MISSING_REASON_CURRENT_OUTLIER
                    else:
                        current_topic_index = global_topic_to_index[int(row.bertopic_topic_id_answer)]
                        current_topic_prior_share = float(prior_probability_vector[current_topic_index])
                        topic_jsd = float(js_divergence_from_onehot(prior_probability_vector, current_topic_index))
                        topic_jsd_norm = float(topic_jsd / max_jsd) if max_jsd > 0.0 else np.nan
                        missing_reason = ""

                rows.append(
                    {
                        "event_id": row.event_id,
                        "question_id": question_id,
                        "bertopic_topic_id_answer": int(row.bertopic_topic_id_answer),
                        "bertopic_topic_name_answer": row.bertopic_topic_name_answer,
                        "bertopic_topic_probability_answer": (
                            float(row.bertopic_topic_probability_answer)
                            if pd.notna(row.bertopic_topic_probability_answer)
                            else np.nan
                        ),
                        "topic_prior_valid_answer_count_bertopic": prior_valid_count,
                        "topic_prior_unique_topic_count_bertopic": prior_unique_topic_count,
                        "topic_prior_outlier_answer_count_bertopic": int(prior_outlier_count),
                        "topic_prior_outlier_ratio_bertopic": prior_outlier_ratio,
                        "topic_prior_entropy_bertopic": prior_entropy,
                        "topic_prior_entropy_bertopic_norm": prior_entropy_norm,
                        "topic_knowledge_diff_bertopic_jsd": topic_jsd,
                        "topic_knowledge_diff_bertopic_jsd_norm": topic_jsd_norm,
                        "topic_knowledge_diff_bertopic_current_topic_prior_share": current_topic_prior_share,
                        "topic_knowledge_diff_bertopic_missing_reason": missing_reason,
                    }
                )

            for topic_id in timestamp_df["bertopic_topic_id_answer"].tolist():
                prior_total_count += 1
                if int(topic_id) == OUTLIER_TOPIC_ID:
                    prior_outlier_count += 1
                else:
                    prior_valid_counter[int(topic_id)] += 1

    return pd.DataFrame.from_records(rows)


def build_summary(
    output_df: pd.DataFrame,
    topic_info_df: pd.DataFrame,
    device: str,
    model_source: str,
) -> dict[str, object]:
    entropy_valid = output_df["topic_prior_entropy_bertopic"].dropna()
    jsd_valid = output_df["topic_knowledge_diff_bertopic_jsd"].dropna()
    valid_topic_count = int((topic_info_df["Topic"] != OUTLIER_TOPIC_ID).sum())
    preview_cols = [
        "question_id",
        "event_id",
        "time_seq",
        "bertopic_topic_id_answer",
        "bertopic_topic_name_answer",
        "topic_prior_valid_answer_count_bertopic",
        "topic_prior_outlier_ratio_bertopic",
        "topic_prior_entropy_bertopic",
        "topic_knowledge_diff_bertopic_jsd",
        "topic_knowledge_diff_bertopic_missing_reason",
    ]
    preview_df = output_df[preview_cols].head(12).copy()

    return {
        "run_ts_utc": utc_now_iso(),
        "input_knowledge_path": str(KNOWLEDGE_PATH),
        "input_answer_path": str(ANSWER_PATH),
        "output_path": str(KNOWLEDGE_PATH),
        "topic_info_path": str(TOPIC_INFO_PATH),
        "algorithm": {
            "topic_model": "BERTopic",
            "embedding_model": MODEL_NAME,
            "embedding_model_source_used": model_source,
            "device_used": device,
            "document_unit": "回答级 answer_text",
            "dimensionality_reduction_model": "TruncatedSVD(n_components=5)",
            "cluster_model": "KMeans(n_clusters=12)",
            "same_timestamp_rule": "同时间戳回答不互为前序，先计算后统一加入前序主题历史。",
            "prior_entropy_formula": "-sum_k p_k ln p_k",
            "topic_diff_formula": "Jensen-Shannon divergence between current one-hot topic vector and prior valid-topic distribution",
            "outlier_handling": "若主题模型产生 -1，则视为离群回答并排除在前序主题熵之外；本次 KMeans 配置通常不会产生 -1。",
            "manual_topic_merge_applied": False,
        },
        "topic_model_counts": {
            "document_count": int(len(output_df)),
            "valid_topic_count_global": valid_topic_count,
            "outlier_document_count": int((output_df["bertopic_topic_id_answer"] == OUTLIER_TOPIC_ID).sum()),
        },
        "prior_entropy_distribution": {
            "valid_rows": int(output_df["topic_prior_entropy_bertopic"].notna().sum()),
            "mean": float(entropy_valid.mean()) if not entropy_valid.empty else None,
            "median": float(entropy_valid.median()) if not entropy_valid.empty else None,
            "p75": float(entropy_valid.quantile(0.75)) if not entropy_valid.empty else None,
            "max": float(entropy_valid.max()) if not entropy_valid.empty else None,
        },
        "topic_diff_distribution": {
            "valid_rows": int(output_df["topic_knowledge_diff_bertopic_jsd"].notna().sum()),
            "mean": float(jsd_valid.mean()) if not jsd_valid.empty else None,
            "median": float(jsd_valid.median()) if not jsd_valid.empty else None,
            "p75": float(jsd_valid.quantile(0.75)) if not jsd_valid.empty else None,
            "max": float(jsd_valid.max()) if not jsd_valid.empty else None,
        },
        "topic_diff_missing_reason_distribution": (
            output_df["topic_knowledge_diff_bertopic_missing_reason"]
            .replace("", "[valid]")
            .value_counts(dropna=False)
            .rename_axis("reason")
            .reset_index(name="row_count")
            .to_dict(orient="records")
        ),
        "topic_info_preview": topic_info_df.head(15).to_dict(orient="records"),
        "preview": preview_df.to_dict(orient="records"),
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    knowledge_df, answer_df = load_inputs()
    model, device, model_source = load_embedding_model()
    docs = answer_df["answer_text"].fillna("").astype(str).tolist()
    embeddings = encode_texts(model, docs)

    topic_model = build_topic_model()
    print("Starting BERTopic fit_transform on flow_run100 answers...")
    topic_ids, topic_probabilities = topic_model.fit_transform(docs, embeddings)
    print("BERTopic fit_transform completed.")
    topic_info_df = topic_name_lookup_df(topic_model)
    topic_info_df.to_csv(TOPIC_INFO_PATH, index=False, encoding="utf-8-sig")

    valid_topic_ids = sorted(int(topic_id) for topic_id in topic_info_df["Topic"].tolist() if int(topic_id) != -1)
    topic_name_lookup = {
        int(row.Topic): str(row.Name)
        for row in topic_info_df.itertuples(index=False)
    }
    topic_metric_df = compute_topic_metrics(
        answer_df=answer_df,
        topic_ids=[int(topic_id) for topic_id in topic_ids],
        topic_probabilities=topic_probabilities,
        valid_topic_ids=valid_topic_ids,
        topic_name_lookup=topic_name_lookup,
    )

    output_df = knowledge_df.drop(
        columns=[column for column in TOPIC_OUTPUT_COLUMNS if column in knowledge_df.columns],
        errors="ignore",
    ).merge(
        topic_metric_df,
        on=["event_id", "question_id"],
        how="left",
        validate="one_to_one",
    )
    output_df = output_df.sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    output_df.to_parquet(KNOWLEDGE_PATH, index=False)

    summary = build_summary(output_df, topic_info_df, device=device, model_source=model_source)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05l_backfill_flow_run100_bertopic_metrics",
                "action": "backfill_bertopic_prior_entropy_and_topic_diff",
                "status": "completed",
                "target_path": str(KNOWLEDGE_PATH),
                "details": (
                    f"rows={len(output_df)}; valid_topics={summary['topic_model_counts']['valid_topic_count_global']}; "
                    f"entropy_valid={summary['prior_entropy_distribution']['valid_rows']}; "
                    f"topic_diff_valid={summary['topic_diff_distribution']['valid_rows']}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05l_backfill_flow_run100_bertopic_metrics",
                "action": "write_bertopic_topic_outputs",
                "status": "completed",
                "target_path": str(TOPIC_INFO_PATH),
                "details": "saved BERTopic topic info table for manual inspection",
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05l_backfill_flow_run100_bertopic_metrics",
                "action": "write_bertopic_metric_summary",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"prior_entropy_mean={summary['prior_entropy_distribution']['mean']}; "
                    f"topic_diff_mean={summary['topic_diff_distribution']['mean']}"
                ),
            },
        ]
    )

    print(f"Backfilled BERTopic metrics written to {KNOWLEDGE_PATH}")
    print(f"Topic info written to {TOPIC_INFO_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
