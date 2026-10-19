from __future__ import annotations

import csv
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
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

ANSWER_PATH = ANALYSIS_DIR / "flow_run100_answer_text_for_topic_clean.csv"
SENTENCE_PATH = ANALYSIS_DIR / "flow_run100_sentence_text_for_topic_clean.csv"
ANSWER_OUTPUT_PATH = ANALYSIS_DIR / "flow_run100_answer_sentence_topic_distribution_k8.csv"
SENTENCE_OUTPUT_PATH = ANALYSIS_DIR / "flow_run100_sentence_topic_labels_k8.csv"
SUMMARY_PATH = AUDIT_DIR / "20i_flow_run100_sentence_bertopic_distribution_summary_k8.json"
TOPIC_INFO_PATH = TABLE_DIR / "20i_flow_run100_sentence_bertopic_topic_info_k8.csv"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"
LOCAL_MODEL_DIR = ROOT_DIR / "models" / "bge-base-zh-v1.5"

MODEL_NAME = "BAAI/bge-base-zh-v1.5"
DEFAULT_FIXED_TOPIC_K = 4
BATCH_SIZE = 64
OUTLIER_TOPIC_ID = -1

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


def import_bertopic():
    original_find_spec = importlib.util.find_spec
    importlib.util.find_spec = lambda name, *args, **kwargs: (  # type: ignore[assignment]
        None if name == "plotly" else original_find_spec(name, *args, **kwargs)
    )
    try:
        from bertopic._bertopic import BERTopic
    finally:
        importlib.util.find_spec = original_find_spec
    return BERTopic


def build_topic_model():
    from hdbscan import HDBSCAN
    from umap import UMAP

    BERTopic = import_bertopic()
    vectorizer_model = CountVectorizer(
        tokenizer=jieba.lcut,
        token_pattern=None,
        lowercase=False,
        min_df=2,
    )
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=20261014,
        low_memory=True,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=20,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
        low_memory=True,
    )
    return topic_model


def get_fixed_topic_k() -> int:
    return int(os.environ.get("FLOW_RUN100_TOPIC_K", DEFAULT_FIXED_TOPIC_K))


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    )
    return normalized.fillna(False).astype(bool)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    answer_df = pd.read_csv(ANSWER_PATH, encoding="utf-8-sig").copy()
    answer_df["question_id"] = answer_df["question_id"].astype(str)
    answer_df["answer_create_time"] = pd.to_datetime(answer_df["answer_create_time"])
    if "has_clean_topic_text" in answer_df.columns:
        answer_df["has_clean_topic_text"] = coerce_bool_series(answer_df["has_clean_topic_text"])
        answer_df = answer_df.loc[answer_df["has_clean_topic_text"]].copy()
    if "question_text" in answer_df.columns and "question_desc" not in answer_df.columns:
        answer_df["question_desc"] = answer_df["question_text"]
    if "focal_answer_text_raw" in answer_df.columns and "answer_text" not in answer_df.columns:
        answer_df["answer_text"] = answer_df["focal_answer_text_raw"]

    question_ids = set(answer_df["question_id"].tolist())
    sentence_df = pd.read_csv(SENTENCE_PATH, encoding="utf-8-sig").copy()
    sentence_df["question_id"] = sentence_df["question_id"].astype(str)
    sentence_df = sentence_df.loc[sentence_df["question_id"].isin(question_ids)].copy()
    if "drop_for_topic_cleaning" in sentence_df.columns:
        sentence_df["drop_for_topic_cleaning"] = coerce_bool_series(sentence_df["drop_for_topic_cleaning"])
        sentence_df = sentence_df.loc[~sentence_df["drop_for_topic_cleaning"]].copy()
    sentence_df["sentence_text_topic_clean"] = sentence_df["sentence_text_topic_clean"].fillna("").astype(str)
    sentence_df = sentence_df.loc[sentence_df["sentence_text_topic_clean"].str.strip() != ""].copy()
    sentence_df["sentence_text"] = sentence_df["sentence_text_topic_clean"]
    sentence_df = sentence_df.sort_values(
        ["question_id", "event_id", "sentence_order"],
        kind="stable",
    ).reset_index(drop=True)
    return answer_df, sentence_df


def reduce_to_fixed_topics(topic_model, docs: list[str], fixed_topic_k: int) -> list[int]:
    topic_model.reduce_topics(docs, nr_topics=fixed_topic_k)
    reduced_topics = [int(topic_id) for topic_id in topic_model.topics_]
    return reduced_topics


def build_sentence_output(
    sentence_df: pd.DataFrame,
    reduced_topics: list[int],
    topic_info_df: pd.DataFrame,
) -> pd.DataFrame:
    sentence_output_df = sentence_df.copy()
    sentence_output_df["sentence_topic_id"] = reduced_topics
    topic_name_lookup = {
        int(row.Topic): str(row.Name)
        for row in topic_info_df.itertuples(index=False)
    }
    sentence_output_df["sentence_topic_name"] = sentence_output_df["sentence_topic_id"].map(
        lambda x: topic_name_lookup.get(int(x), "")
    )
    return sentence_output_df


def build_answer_topic_distribution(
    answer_df: pd.DataFrame,
    sentence_topic_df: pd.DataFrame,
    valid_topic_ids: list[int],
    fixed_topic_k: int,
) -> pd.DataFrame:
    answer_records: list[dict[str, object]] = []
    topic_id_set = set(valid_topic_ids)

    for row in answer_df.itertuples(index=False):
        sentence_slice = sentence_topic_df.loc[sentence_topic_df["event_id"] == row.event_id].copy()
        total_sentence_count = int(len(sentence_slice))
        valid_slice = sentence_slice.loc[sentence_slice["sentence_topic_id"].isin(topic_id_set)].copy()
        noise_count = int((sentence_slice["sentence_topic_id"] == OUTLIER_TOPIC_ID).sum())
        valid_sentence_count = int(len(valid_slice))
        has_valid_topic = valid_sentence_count > 0

        if has_valid_topic:
            counts = (
                valid_slice["sentence_topic_id"]
                .value_counts()
                .reindex(valid_topic_ids, fill_value=0)
                .astype(int)
            )
            distribution = (counts / valid_sentence_count).astype(float)
            topic_distribution_json = json.dumps(
                {str(topic_id): float(distribution.loc[topic_id]) for topic_id in valid_topic_ids},
                ensure_ascii=False,
            )
            topic_coverage = int((counts > 0).sum())
            dominant_topic_id = int(counts.idxmax())
            dominant_topic_share = float(distribution.loc[dominant_topic_id])
        else:
            topic_distribution_json = None
            topic_coverage = 0
            dominant_topic_id = None
            dominant_topic_share = np.nan

        noise_sentence_share = float(noise_count / total_sentence_count) if total_sentence_count else np.nan
        answer_records.append(
            {
                "sample_layer": row.sample_layer,
                "question_id": row.question_id,
                "event_id": row.event_id,
                "user_id": row.user_id,
                "time_seq": int(row.time_seq),
                "seq": int(row.seq),
                "answer_create_time": row.answer_create_time,
                "question_text": row.question_desc,
                "focal_answer_text": row.answer_text,
                "total_sentence_count": total_sentence_count,
                "valid_topic_sentence_count": valid_sentence_count,
                "noise_sentence_count": noise_count,
                "noise_sentence_share": noise_sentence_share,
                "has_valid_topic": bool(has_valid_topic),
                "topic_coverage": topic_coverage,
                "topic_distribution_json": topic_distribution_json,
                "dominant_topic_id": dominant_topic_id,
                "dominant_topic_share": dominant_topic_share,
                "fixed_topic_k": fixed_topic_k,
            }
        )

    answer_topic_df = pd.DataFrame.from_records(answer_records)
    answer_topic_df = answer_topic_df.sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    return answer_topic_df


def build_summary(
    answer_topic_df: pd.DataFrame,
    sentence_topic_df: pd.DataFrame,
    topic_info_df: pd.DataFrame,
    fixed_topic_k: int,
    device: str,
    model_source: str,
) -> dict[str, object]:
    valid_topic_count = int((topic_info_df["Topic"] != OUTLIER_TOPIC_ID).sum())
    return {
        "run_ts_utc": utc_now_iso(),
        "scope": "flow_run100_sentence_level_bertopic_distribution",
        "inputs": {
            "answer_path": str(ANSWER_PATH),
            "sentence_path": str(SENTENCE_PATH),
        },
        "outputs": {
            "answer_topic_output_path": str(ANSWER_OUTPUT_PATH),
            "sentence_topic_output_path": str(SENTENCE_OUTPUT_PATH),
            "topic_info_path": str(TOPIC_INFO_PATH),
        },
        "algorithm": {
            "topic_model": "BERTopic",
            "embedding_model": MODEL_NAME,
            "embedding_model_source_used": model_source,
            "device_used": device,
            "cluster_backend": cluster_backend,
            "document_unit_for_training": "句子级 sentence_text_topic_clean",
            "answer_distribution_unit": "回答内句子主题计数 / 有效主题句数",
            "fixed_topic_k_requested": fixed_topic_k,
            "topic_reduction_applied": topic_reduction_applied,
            "topic_reduction_strategy": (
                f"reduce_topics -> {fixed_topic_k}"
                if topic_reduction_applied
                else ("KMeans fixed n_clusters" if cluster_backend == "kmeans" else "none")
            ),
            "umap_n_neighbors": DEFAULT_UMAP_NEIGHBORS,
            "umap_n_components": DEFAULT_UMAP_COMPONENTS,
            "hdbscan_min_cluster_size": (
                None if cluster_backend == "kmeans" else DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE
            ),
            "hdbscan_min_samples": (
                None if cluster_backend == "kmeans" else DEFAULT_HDBSCAN_MIN_SAMPLES
            ),
            "noise_rule": "HDBSCAN 的 -1 视为噪声句，不作为可解释主题；若整条回答没有有效主题，则主题分布记缺失。",
            "saved_flags": ["topic_coverage", "noise_sentence_share", "has_valid_topic"],
            "approximate_distribution_used": False,
        },
        "counts": {
            "answer_rows": int(len(answer_topic_df)),
            "sentence_rows": int(len(sentence_topic_df)),
            "answers_with_valid_topic": int(answer_topic_df["has_valid_topic"].sum()),
            "answers_without_valid_topic": int((~answer_topic_df["has_valid_topic"]).sum()),
            "global_valid_topic_count_after_reduction": valid_topic_count,
            "global_noise_sentence_count": int((sentence_topic_df["sentence_topic_id"] == OUTLIER_TOPIC_ID).sum()),
        },
        "distribution_summary": {
            "topic_coverage_mean": float(answer_topic_df["topic_coverage"].mean()),
            "topic_coverage_max": int(answer_topic_df["topic_coverage"].max()),
            "noise_sentence_share_mean": float(answer_topic_df["noise_sentence_share"].dropna().mean()),
        },
        "topic_info_preview": topic_info_df.head(15).to_dict(orient="records"),
        "answer_preview": answer_topic_df.head(10).assign(answer_create_time=lambda d: d["answer_create_time"].astype(str)).to_dict(orient="records"),
    }


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    fixed_topic_k = get_fixed_topic_k()
    answer_df, sentence_df = load_inputs()

    texts = sentence_df["sentence_text"].fillna("").astype(str).tolist()
    print(f"Loaded {len(answer_df)} answers and {len(sentence_df)} sentences.")
    model, device, model_source = load_embedding_model()
    print(f"Embedding model ready on {device}: {model_source}")
    embeddings = encode_texts(model, texts)
    print("Sentence embeddings completed.")

    topic_model = build_topic_model()
    print("BERTopic model built. Starting fit_transform...")
    topic_model.fit_transform(texts, embeddings)
    print("BERTopic fit_transform completed. Starting topic reduction...")
    reduced_topics = reduce_to_fixed_topics(topic_model, texts, fixed_topic_k=fixed_topic_k)
    print("Topic reduction completed.")
    topic_info_df = topic_model.get_topic_info().copy()
    topic_info_df.to_csv(TOPIC_INFO_PATH, index=False, encoding="utf-8-sig")

    sentence_topic_df = build_sentence_output(sentence_df, reduced_topics, topic_info_df)
    valid_topic_ids = sorted(int(topic_id) for topic_id in topic_info_df["Topic"].tolist() if int(topic_id) != OUTLIER_TOPIC_ID)
    answer_topic_df = build_answer_topic_distribution(
        answer_df=answer_df,
        sentence_topic_df=sentence_topic_df,
        valid_topic_ids=valid_topic_ids,
        fixed_topic_k=fixed_topic_k,
    )

    sentence_topic_save_df = sentence_topic_df.copy()
    sentence_topic_save_df.to_csv(SENTENCE_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    answer_topic_save_df = answer_topic_df.copy()
    answer_topic_save_df["answer_create_time"] = answer_topic_save_df["answer_create_time"].astype(str)
    answer_topic_save_df.to_csv(ANSWER_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary = build_summary(
        answer_topic_df=answer_topic_df,
        sentence_topic_df=sentence_topic_df,
        topic_info_df=topic_info_df,
        fixed_topic_k=fixed_topic_k,
        device=device,
        model_source=model_source,
    )
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05n_run_flow_run100_sentence_bertopic_distribution",
                "action": "write_answer_level_sentence_topic_distribution",
                "status": "completed",
                "target_path": str(ANSWER_OUTPUT_PATH),
                "details": (
                    f"answer_rows={summary['counts']['answer_rows']}; "
                    f"answers_with_valid_topic={summary['counts']['answers_with_valid_topic']}; "
                    f"fixed_topic_k={fixed_topic_k}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05n_run_flow_run100_sentence_bertopic_distribution",
                "action": "write_sentence_level_topic_labels",
                "status": "completed",
                "target_path": str(SENTENCE_OUTPUT_PATH),
                "details": f"sentence_rows={summary['counts']['sentence_rows']}",
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05n_run_flow_run100_sentence_bertopic_distribution",
                "action": "write_sentence_bertopic_summary",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"valid_topic_count={summary['counts']['global_valid_topic_count_after_reduction']}; "
                    f"noise_sentence_count={summary['counts']['global_noise_sentence_count']}"
                ),
            },
        ]
    )

    print(f"Answer topic distribution written to {ANSWER_OUTPUT_PATH}")
    print(f"Sentence topic labels written to {SENTENCE_OUTPUT_PATH}")
    print(f"Topic info written to {TOPIC_INFO_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
