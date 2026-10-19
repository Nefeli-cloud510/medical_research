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

INPUT_PATH = ANALYSIS_DIR / "flow_run100_answer_text_features.parquet"
OUTPUT_CSV_PATH = ANALYSIS_DIR / "flow_run100_knowledge_metrics_bge_mean_prior_similarity.csv"
SUMMARY_PATH = AUDIT_DIR / "20h_flow_run100_mean_prior_similarity_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"
LOCAL_MODEL_DIR = ROOT_DIR / "models" / "bge-base-zh-v1.5"

MODEL_NAME = "BAAI/bge-base-zh-v1.5"
BATCH_SIZE = 64
EMBEDDING_DIM = 768
MISSING_REASON_NO_PRIOR = "no_strict_prior"

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
        group_positions = group_df.index.to_numpy()
        group_embeddings = embeddings[group_positions]

        prior_vector_sum = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        prior_answer_count = 0

        for _, timestamp_df in group_df.groupby("answer_create_time", sort=False):
            timestamp_df = timestamp_df.sort_values(["time_seq", "seq"], kind="stable")
            timestamp_positions = timestamp_df.index.to_numpy()
            timestamp_offsets = np.searchsorted(group_positions, timestamp_positions)
            timestamp_embeddings = group_embeddings[timestamp_offsets]

            for row, current_embedding in zip(timestamp_df.itertuples(index=False), timestamp_embeddings):
                if prior_answer_count == 0:
                    mean_prior_similarity = np.nan
                    knowledge_differentiation = np.nan
                    missing_reason = MISSING_REASON_NO_PRIOR
                    prior_mean_vector_norm = np.nan
                else:
                    prior_mean_vector = prior_vector_sum / prior_answer_count
                    prior_mean_vector_norm = float(np.linalg.norm(prior_mean_vector))
                    mean_prior_similarity = float(np.dot(current_embedding, prior_mean_vector))
                    knowledge_differentiation = float(1.0 - mean_prior_similarity)
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
                        "strict_prior_mean_vector_norm": prior_mean_vector_norm,
                        "mean_semantic_similarity_to_prior_answers": mean_prior_similarity,
                        "knowledge_differentiation_bge_mean_prior_similarity": knowledge_differentiation,
                        "knowledge_differentiation_mean_prior_missing_reason": missing_reason,
                        "question_text": row.question_desc,
                        "focal_answer_text": row.answer_text,
                        "all_prior_answers_text": row.strict_prior_text_concat,
                        "strict_prior_event_ids_json": row.strict_prior_event_ids_json,
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
) -> dict[str, object]:
    valid_series = output_df["knowledge_differentiation_bge_mean_prior_similarity"].dropna()
    similarity_series = output_df["mean_semantic_similarity_to_prior_answers"].dropna()
    preview_df = output_df.head(10).copy()
    if "answer_create_time" in preview_df.columns:
        preview_df["answer_create_time"] = preview_df["answer_create_time"].astype(str)

    return {
        "run_ts_utc": utc_now_iso(),
        "scope": "flow_run100_mean_prior_similarity_knowledge_differentiation",
        "input_path": str(INPUT_PATH),
        "output_csv_path": str(OUTPUT_CSV_PATH),
        "algorithm": {
            "model_name": MODEL_NAME,
            "model_source_used": model_source,
            "local_model_dir": str(LOCAL_MODEL_DIR),
            "embedding_dim": EMBEDDING_DIM,
            "device_used": device,
            "normalize_embeddings": True,
            "mean_similarity_formula": "S_ij = (1/n_ij) * sum_{r in P_ij} cos(e_ij, e_ir)",
            "knowledge_differentiation_formula": "D_ij = 1 - S_ij",
            "implementation_note": "由于回答向量已归一化，平均相似度通过当前向量与未归一化前序平均向量的点积实现。",
            "same_timestamp_rule": "同时间戳回答先计算指标，再统一加入前序回答集合。",
            "no_prior_rule": "n_ij = 0 时平均相似度和知识分化度均记缺失。",
            "input_text_unit": "整条回答 answer_text，不附加问题文本进入编码。",
        },
        "counts": {
            "total_rows": int(len(output_df)),
            "valid_rows": int(output_df["knowledge_differentiation_bge_mean_prior_similarity"].notna().sum()),
            "missing_rows": int(output_df["knowledge_differentiation_bge_mean_prior_similarity"].isna().sum()),
            "valid_rate": round(
                float(output_df["knowledge_differentiation_bge_mean_prior_similarity"].notna().mean()),
                6,
            ),
        },
        "mean_similarity_distribution": {
            "min": float(similarity_series.min()) if not similarity_series.empty else None,
            "p25": float(similarity_series.quantile(0.25)) if not similarity_series.empty else None,
            "median": float(similarity_series.median()) if not similarity_series.empty else None,
            "p75": float(similarity_series.quantile(0.75)) if not similarity_series.empty else None,
            "max": float(similarity_series.max()) if not similarity_series.empty else None,
            "mean": float(similarity_series.mean()) if not similarity_series.empty else None,
        },
        "knowledge_diff_distribution": {
            "min": float(valid_series.min()) if not valid_series.empty else None,
            "p25": float(valid_series.quantile(0.25)) if not valid_series.empty else None,
            "median": float(valid_series.median()) if not valid_series.empty else None,
            "p75": float(valid_series.quantile(0.75)) if not valid_series.empty else None,
            "max": float(valid_series.max()) if not valid_series.empty else None,
            "mean": float(valid_series.mean()) if not valid_series.empty else None,
        },
        "missing_reason_distribution": (
            output_df["knowledge_differentiation_mean_prior_missing_reason"]
            .replace("", "[valid]")
            .value_counts(dropna=False)
            .rename_axis("reason")
            .reset_index(name="row_count")
            .to_dict(orient="records")
        ),
        "preview": preview_df.to_dict(orient="records"),
    }


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH).copy()
    df["answer_create_time"] = pd.to_datetime(df["answer_create_time"])
    df = df.sort_values(
        ["question_id", "answer_create_time", "time_seq", "seq"],
        kind="stable",
    ).reset_index(drop=True)

    texts = df["answer_text"].fillna("").astype(str).tolist()
    model, device, model_source = load_model()
    embeddings = encode_answers(model, texts)
    output_df = compute_metrics(df, embeddings)

    csv_df = output_df.copy()
    csv_df["answer_create_time"] = csv_df["answer_create_time"].astype(str)
    csv_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    summary = build_summary(output_df, device=device, model_source=model_source)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05m_compute_flow_run100_mean_prior_similarity_csv",
                "action": "compute_mean_prior_similarity_knowledge_differentiation_csv",
                "status": "completed",
                "target_path": str(OUTPUT_CSV_PATH),
                "details": (
                    f"rows={len(output_df)}; model={MODEL_NAME}; device={device}; "
                    f"valid_rows={summary['counts']['valid_rows']}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05m_compute_flow_run100_mean_prior_similarity_csv",
                "action": "write_mean_prior_similarity_summary",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"missing_rows={summary['counts']['missing_rows']}; "
                    f"valid_rate={summary['counts']['valid_rate']}"
                ),
            },
        ]
    )

    print(f"Mean-prior similarity CSV written to {OUTPUT_CSV_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
