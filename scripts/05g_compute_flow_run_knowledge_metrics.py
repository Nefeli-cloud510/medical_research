from __future__ import annotations

import csv
import json
import os
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

DEFAULT_INPUT_PATH = ANALYSIS_DIR / "flow_run_answer_text_features.parquet"
DEFAULT_OUTPUT_PATH = ANALYSIS_DIR / "flow_run_knowledge_metrics_bge.parquet"
DEFAULT_SUMMARY_PATH = AUDIT_DIR / "20_flow_run_knowledge_metrics_bge_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"
LOCAL_MODEL_DIR = ROOT_DIR / "models" / "bge-base-zh-v1.5"

MODEL_NAME = "BAAI/bge-base-zh-v1.5"
BATCH_SIZE = 64
EMBEDDING_DIM = 768
MISSING_REASON_NO_PRIOR = "no_strict_prior"
MISSING_REASON_ZERO_CENTROID = "prior_centroid_zero_norm"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]


def get_input_path() -> Path:
    return Path(os.environ.get("KNOWLEDGE_METRICS_INPUT_PATH", str(DEFAULT_INPUT_PATH)))


def get_output_path() -> Path:
    return Path(os.environ.get("KNOWLEDGE_METRICS_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH)))


def get_summary_path() -> Path:
    return Path(os.environ.get("KNOWLEDGE_METRICS_SUMMARY_PATH", str(DEFAULT_SUMMARY_PATH)))


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


def choose_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model() -> tuple[SentenceTransformer, str]:
    device = choose_device()
    model_source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_NAME
    model = SentenceTransformer(model_source, device=device)
    return model, device


def encode_answers(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32, copy=False)


def compute_metrics(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    output_rows: list[dict[str, object]] = []

    for question_id, group_index in df.groupby("question_id", sort=False).groups.items():
        group_df = df.loc[group_index].sort_values(
            ["answer_create_time", "time_seq", "seq"],
            kind="stable",
        )
        group_embeddings = embeddings[group_df.index.to_numpy()]

        prior_vector_sum = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        prior_answer_count = 0

        timestamp_groups = list(group_df.groupby("answer_create_time", sort=False))
        for _, timestamp_df in timestamp_groups:
            timestamp_df = timestamp_df.sort_values(["time_seq", "seq"], kind="stable")
            timestamp_embeddings = group_embeddings[
                timestamp_df.index.to_numpy() - group_df.index.min()
            ]

            for row, current_embedding in zip(timestamp_df.itertuples(index=False), timestamp_embeddings):
                centroid_norm = float(np.linalg.norm(prior_vector_sum / prior_answer_count)) if prior_answer_count else np.nan
                if prior_answer_count == 0:
                    cosine_similarity = np.nan
                    knowledge_differentiation = np.nan
                    missing_reason = MISSING_REASON_NO_PRIOR
                else:
                    centroid = prior_vector_sum / prior_answer_count
                    centroid_norm = float(np.linalg.norm(centroid))
                    if centroid_norm <= 0.0:
                        cosine_similarity = np.nan
                        knowledge_differentiation = np.nan
                        missing_reason = MISSING_REASON_ZERO_CENTROID
                    else:
                        cosine_similarity = float(np.dot(current_embedding, centroid) / centroid_norm)
                        knowledge_differentiation = float(1.0 - cosine_similarity)
                        missing_reason = ""

                output_rows.append(
                    {
                        "sample_layer": row.sample_layer,
                        "question_id": question_id,
                        "event_id": row.event_id,
                        "user_id": row.user_id,
                        "time_seq": int(row.time_seq),
                        "seq": int(row.seq),
                        "answer_create_time": row.answer_create_time,
                        "peer_crowding_prior_answer_count": int(row.strict_prior_answer_count_current),
                        "peer_crowding_prior_total_char_count": int(row.strict_prior_total_char_count_current),
                        "current_answer_char_count": int(row.current_answer_char_count),
                        "knowledge_diff_model": MODEL_NAME,
                        "current_embedding_norm": float(np.linalg.norm(current_embedding)),
                        "strict_prior_centroid_norm": centroid_norm,
                        "knowledge_diff_cosine_similarity_to_prior_centroid": cosine_similarity,
                        "knowledge_differentiation_bge_centroid": knowledge_differentiation,
                        "knowledge_differentiation_missing_reason": missing_reason,
                    }
                )

            prior_vector_sum = prior_vector_sum + timestamp_embeddings.sum(axis=0).astype(np.float32, copy=False)
            prior_answer_count += int(len(timestamp_df))

    output_df = pd.DataFrame.from_records(output_rows)
    output_df = output_df.sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    return output_df


def build_summary(
    output_df: pd.DataFrame,
    device: str,
    model_source: str,
    input_path: Path,
    output_path: Path,
) -> dict[str, object]:
    total_rows = int(len(output_df))
    valid_rows = int(output_df["knowledge_differentiation_bge_centroid"].notna().sum())
    missing_rows = total_rows - valid_rows
    missing_reason_distribution = (
        output_df["knowledge_differentiation_missing_reason"]
        .replace("", "[valid]")
        .value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="row_count")
        .to_dict(orient="records")
    )
    valid_metric_series = output_df["knowledge_differentiation_bge_centroid"].dropna()
    preview_df = output_df.head(10).copy()
    if "answer_create_time" in preview_df.columns:
        preview_df["answer_create_time"] = preview_df["answer_create_time"].astype(str)

    return {
        "run_ts_utc": utc_now_iso(),
        "scope": "flow_run_knowledge_metrics_bge_centroid",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "algorithm": {
            "model_name": MODEL_NAME,
            "model_source_used": model_source,
            "local_model_dir": str(LOCAL_MODEL_DIR),
            "embedding_dim": EMBEDDING_DIM,
            "device_used": device,
            "normalize_embeddings": True,
            "crowding_metrics": [
                "peer_crowding_prior_answer_count",
                "peer_crowding_prior_total_char_count",
            ],
            "knowledge_differentiation_formula": "1 - cos(e_j, mean(P_j))",
            "same_timestamp_rule": "同时间戳回答先计算指标，再统一加入前序向量和。",
            "no_prior_rule": "n_j = 0 时知识分化度记为缺失值，不记为 0。",
        },
        "counts": {
            "total_rows": total_rows,
            "valid_knowledge_diff_rows": valid_rows,
            "missing_knowledge_diff_rows": missing_rows,
            "valid_rate": round(valid_rows / total_rows, 6) if total_rows else 0.0,
        },
        "knowledge_diff_distribution": {
            "min": float(valid_metric_series.min()) if not valid_metric_series.empty else None,
            "p25": float(valid_metric_series.quantile(0.25)) if not valid_metric_series.empty else None,
            "median": float(valid_metric_series.median()) if not valid_metric_series.empty else None,
            "p75": float(valid_metric_series.quantile(0.75)) if not valid_metric_series.empty else None,
            "max": float(valid_metric_series.max()) if not valid_metric_series.empty else None,
            "mean": float(valid_metric_series.mean()) if not valid_metric_series.empty else None,
        },
        "missing_reason_distribution": missing_reason_distribution,
        "preview": preview_df.to_dict(orient="records"),
    }


def main() -> None:
    ensure_directories()
    input_path = get_input_path()
    output_path = get_output_path()
    summary_path = get_summary_path()

    df = pd.read_parquet(input_path).copy()
    df["answer_create_time"] = pd.to_datetime(df["answer_create_time"])
    df = df.sort_values(["question_id", "answer_create_time", "time_seq", "seq"], kind="stable").reset_index(drop=True)

    texts = df["answer_text"].fillna("").astype(str).tolist()
    model, device = load_model()
    model_source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_NAME
    embeddings = encode_answers(model, texts)
    output_df = compute_metrics(df, embeddings)

    output_df.to_parquet(output_path, index=False)
    summary = build_summary(
        output_df,
        device=device,
        model_source=model_source,
        input_path=input_path,
        output_path=output_path,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05g_compute_flow_run_knowledge_metrics",
                "action": "compute_peer_crowding_and_bge_centroid_knowledge_differentiation",
                "status": "completed",
                "target_path": str(output_path),
                "details": (
                    f"rows={len(output_df)}; model={MODEL_NAME}; device={device}; "
                    f"valid_rows={summary['counts']['valid_knowledge_diff_rows']}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05g_compute_flow_run_knowledge_metrics",
                "action": "write_knowledge_metric_summary",
                "status": "completed",
                "target_path": str(summary_path),
                "details": (
                    f"missing_rows={summary['counts']['missing_knowledge_diff_rows']}; "
                    f"valid_rate={summary['counts']['valid_rate']}"
                ),
            },
        ]
    )

    print(f"Knowledge metrics written to {output_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
