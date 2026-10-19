from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

KNOWLEDGE_PATH = ANALYSIS_DIR / "flow_run100_knowledge_metrics_bge.parquet"
ANSWER_PATH = ANALYSIS_DIR / "flow_run100_answer_text_features.parquet"
SENTENCE_PATH = ANALYSIS_DIR / "flow_run_sentence_text_features.parquet"
SUMMARY_PATH = AUDIT_DIR / "20c_flow_run100_nearest_and_sentence_metric_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"
LOCAL_MODEL_DIR = ROOT_DIR / "models" / "bge-base-zh-v1.5"

MODEL_NAME = "BAAI/bge-base-zh-v1.5"
BATCH_SIZE = 64
MISSING_REASON_NO_PRIOR = "no_strict_prior"
MISSING_REASON_NO_CURRENT_SENTENCE = "no_current_sentences"
MISSING_REASON_NO_PRIOR_SENTENCE = "no_prior_sentences"

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


def load_model() -> tuple[SentenceTransformer, str, str]:
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


def prepare_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    knowledge_df = pd.read_parquet(KNOWLEDGE_PATH).copy()
    answer_df = pd.read_parquet(ANSWER_PATH).copy()
    sentence_df = pd.read_parquet(SENTENCE_PATH).copy()

    knowledge_df["question_id"] = knowledge_df["question_id"].astype(str)
    answer_df["question_id"] = answer_df["question_id"].astype(str)
    sentence_df["question_id"] = sentence_df["question_id"].astype(str)

    answer_df["answer_create_time"] = pd.to_datetime(answer_df["answer_create_time"])
    knowledge_df["answer_create_time"] = pd.to_datetime(knowledge_df["answer_create_time"])

    answer_df = answer_df.sort_values(
        ["question_id", "answer_create_time", "time_seq", "seq"],
        kind="stable",
    ).reset_index(drop=True)

    valid_event_ids = set(answer_df["event_id"].tolist())
    valid_question_ids = set(answer_df["question_id"].tolist())
    sentence_df = sentence_df.loc[
        sentence_df["event_id"].isin(valid_event_ids)
        & sentence_df["question_id"].isin(valid_question_ids)
    ].copy()
    sentence_df = sentence_df.sort_values(
        ["question_id", "event_id", "sentence_order"],
        kind="stable",
    ).reset_index(drop=True)

    return knowledge_df, answer_df, sentence_df


def build_embedding_lookup(keys: list[str], embeddings: np.ndarray) -> dict[str, np.ndarray]:
    return {key: embeddings[idx] for idx, key in enumerate(keys)}


def compute_answer_nearest_metrics(
    answer_df: pd.DataFrame,
    answer_embedding_lookup: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for question_id, group_df in answer_df.groupby("question_id", sort=False):
        group_df = group_df.sort_values(
            ["answer_create_time", "time_seq", "seq"],
            kind="stable",
        )

        prior_answer_matrix = np.empty((0, 768), dtype=np.float32)

        for _, timestamp_df in group_df.groupby("answer_create_time", sort=False):
            timestamp_df = timestamp_df.sort_values(["time_seq", "seq"], kind="stable")
            current_event_ids = timestamp_df["event_id"].tolist()
            current_answer_matrix = np.vstack([answer_embedding_lookup[event_id] for event_id in current_event_ids])

            for row, current_embedding in zip(timestamp_df.itertuples(index=False), current_answer_matrix):
                if prior_answer_matrix.shape[0] == 0:
                    nearest_cosine = np.nan
                    nearest_distance = np.nan
                    missing_reason = MISSING_REASON_NO_PRIOR
                else:
                    cosine_scores = prior_answer_matrix @ current_embedding
                    nearest_cosine = float(np.max(cosine_scores))
                    nearest_distance = float(1.0 - nearest_cosine)
                    missing_reason = ""

                rows.append(
                    {
                        "event_id": row.event_id,
                        "question_id": question_id,
                        "knowledge_diff_nearest_prior_cosine": nearest_cosine,
                        "knowledge_differentiation_bge_nearest_prior": nearest_distance,
                        "knowledge_diff_nearest_prior_missing_reason": missing_reason,
                    }
                )

            prior_answer_matrix = np.vstack([prior_answer_matrix, current_answer_matrix])

    return pd.DataFrame.from_records(rows)


def compute_sentence_metrics(
    answer_df: pd.DataFrame,
    sentence_df: pd.DataFrame,
    sentence_embedding_lookup: dict[str, np.ndarray],
) -> pd.DataFrame:
    sentence_group_lookup = {
        event_id: group.sort_values("sentence_order", kind="stable")
        for event_id, group in sentence_df.groupby("event_id", sort=False)
    }
    rows: list[dict[str, object]] = []

    for question_id, group_df in answer_df.groupby("question_id", sort=False):
        group_df = group_df.sort_values(
            ["answer_create_time", "time_seq", "seq"],
            kind="stable",
        )
        prior_sentence_matrix = np.empty((0, 768), dtype=np.float32)

        for _, timestamp_df in group_df.groupby("answer_create_time", sort=False):
            timestamp_df = timestamp_df.sort_values(["time_seq", "seq"], kind="stable")
            batch_sentence_matrices: list[np.ndarray] = []

            for row in timestamp_df.itertuples(index=False):
                current_sentence_df = sentence_group_lookup.get(row.event_id)
                if current_sentence_df is None or current_sentence_df.empty:
                    sentence_mean_distance = np.nan
                    sentence_count = 0
                    prior_sentence_count = int(prior_sentence_matrix.shape[0])
                    missing_reason = MISSING_REASON_NO_CURRENT_SENTENCE
                    current_sentence_matrix = np.empty((0, 768), dtype=np.float32)
                else:
                    sentence_ids = current_sentence_df["sentence_id"].tolist()
                    current_sentence_matrix = np.vstack(
                        [sentence_embedding_lookup[sentence_id] for sentence_id in sentence_ids]
                    )
                    sentence_count = int(current_sentence_matrix.shape[0])
                    prior_sentence_count = int(prior_sentence_matrix.shape[0])

                    if prior_sentence_count == 0:
                        sentence_mean_distance = np.nan
                        missing_reason = MISSING_REASON_NO_PRIOR_SENTENCE
                    else:
                        cosine_matrix = current_sentence_matrix @ prior_sentence_matrix.T
                        nearest_prior_cosine_per_sentence = cosine_matrix.max(axis=1)
                        sentence_distances = 1.0 - nearest_prior_cosine_per_sentence
                        sentence_mean_distance = float(sentence_distances.mean())
                        missing_reason = ""

                rows.append(
                    {
                        "event_id": row.event_id,
                        "question_id": question_id,
                        "current_sentence_count": sentence_count,
                        "strict_prior_sentence_count": prior_sentence_count,
                        "knowledge_differentiation_bge_sentence_mean": sentence_mean_distance,
                        "knowledge_diff_sentence_missing_reason": missing_reason,
                    }
                )
                batch_sentence_matrices.append(current_sentence_matrix)

            if batch_sentence_matrices:
                non_empty = [matrix for matrix in batch_sentence_matrices if matrix.shape[0] > 0]
                if non_empty:
                    prior_sentence_matrix = np.vstack([prior_sentence_matrix, *non_empty])

    return pd.DataFrame.from_records(rows)


def build_summary(output_df: pd.DataFrame, device: str, model_source: str) -> dict[str, object]:
    nearest_valid = output_df["knowledge_differentiation_bge_nearest_prior"].dropna()
    sentence_valid = output_df["knowledge_differentiation_bge_sentence_mean"].dropna()
    preview_df = output_df.head(10).copy()
    if "answer_create_time" in preview_df.columns:
        preview_df["answer_create_time"] = preview_df["answer_create_time"].astype(str)

    return {
        "run_ts_utc": utc_now_iso(),
        "input_knowledge_path": str(KNOWLEDGE_PATH),
        "input_answer_path": str(ANSWER_PATH),
        "input_sentence_path": str(SENTENCE_PATH),
        "output_path": str(KNOWLEDGE_PATH),
        "algorithm": {
            "model_name": MODEL_NAME,
            "model_source_used": model_source,
            "device_used": device,
            "nearest_prior_formula": "1 - max cos(e_i, e_j), j in P_i",
            "sentence_mean_formula": "(1/m) * sum_l [1 - max cos(v_sl, u)], u in U_i",
            "same_timestamp_rule": "同时间戳回答先计算，再统一加入前序回答/前序句子历史。",
        },
        "counts": {
            "rows": int(len(output_df)),
            "nearest_valid_rows": int(output_df["knowledge_differentiation_bge_nearest_prior"].notna().sum()),
            "sentence_valid_rows": int(output_df["knowledge_differentiation_bge_sentence_mean"].notna().sum()),
        },
        "nearest_distribution": {
            "mean": float(nearest_valid.mean()) if not nearest_valid.empty else None,
            "median": float(nearest_valid.median()) if not nearest_valid.empty else None,
            "p75": float(nearest_valid.quantile(0.75)) if not nearest_valid.empty else None,
            "max": float(nearest_valid.max()) if not nearest_valid.empty else None,
        },
        "sentence_distribution": {
            "mean": float(sentence_valid.mean()) if not sentence_valid.empty else None,
            "median": float(sentence_valid.median()) if not sentence_valid.empty else None,
            "p75": float(sentence_valid.quantile(0.75)) if not sentence_valid.empty else None,
            "max": float(sentence_valid.max()) if not sentence_valid.empty else None,
        },
        "nearest_missing_reason_distribution": (
            output_df["knowledge_diff_nearest_prior_missing_reason"]
            .replace("", "[valid]")
            .value_counts(dropna=False)
            .rename_axis("reason")
            .reset_index(name="row_count")
            .to_dict(orient="records")
        ),
        "sentence_missing_reason_distribution": (
            output_df["knowledge_diff_sentence_missing_reason"]
            .replace("", "[valid]")
            .value_counts(dropna=False)
            .rename_axis("reason")
            .reset_index(name="row_count")
            .to_dict(orient="records")
        ),
        "preview": preview_df.to_dict(orient="records"),
    }


def main() -> None:
    knowledge_df, answer_df, sentence_df = prepare_inputs()
    model, device, model_source = load_model()

    answer_embeddings = encode_texts(model, answer_df["answer_text"].fillna("").astype(str).tolist())
    sentence_embeddings = encode_texts(model, sentence_df["sentence_text"].fillna("").astype(str).tolist())

    answer_embedding_lookup = build_embedding_lookup(answer_df["event_id"].tolist(), answer_embeddings)
    sentence_embedding_lookup = build_embedding_lookup(sentence_df["sentence_id"].tolist(), sentence_embeddings)

    nearest_df = compute_answer_nearest_metrics(answer_df, answer_embedding_lookup)
    sentence_metric_df = compute_sentence_metrics(answer_df, sentence_df, sentence_embedding_lookup)

    output_df = knowledge_df.merge(
        nearest_df,
        on=["event_id", "question_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        sentence_metric_df,
        on=["event_id", "question_id"],
        how="left",
        validate="one_to_one",
    )
    output_df = output_df.sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    output_df.to_parquet(KNOWLEDGE_PATH, index=False)

    summary = build_summary(output_df, device=device, model_source=model_source)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05h_backfill_flow_run100_nearest_and_sentence_metrics",
                "action": "backfill_nearest_prior_and_sentence_mean_metrics",
                "status": "completed",
                "target_path": str(KNOWLEDGE_PATH),
                "details": (
                    f"rows={len(output_df)}; nearest_valid={summary['counts']['nearest_valid_rows']}; "
                    f"sentence_valid={summary['counts']['sentence_valid_rows']}; model={MODEL_NAME}; device={device}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05h_backfill_flow_run100_nearest_and_sentence_metrics",
                "action": "write_nearest_and_sentence_metric_summary",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"nearest_mean={summary['nearest_distribution']['mean']}; "
                    f"sentence_mean={summary['sentence_distribution']['mean']}"
                ),
            },
        ]
    )

    print(f"Backfilled metrics written to {KNOWLEDGE_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
