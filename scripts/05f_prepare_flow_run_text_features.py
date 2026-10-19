from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import jieba  # type: ignore
except ImportError:  # pragma: no cover
    jieba = None

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

ANSWER_SAMPLE_PATH = ANALYSIS_DIR / "major5_stratified_question_samples_answers.parquet"
SENTENCE_TABLE_PATH = ANALYSIS_DIR / "major5_answer_sentences.parquet"
ANSWER_OUTPUT_PATH = ANALYSIS_DIR / "flow_run_answer_text_features.parquet"
SENTENCE_OUTPUT_PATH = ANALYSIS_DIR / "flow_run_sentence_text_features.parquet"
SUMMARY_PATH = AUDIT_DIR / "19_flow_run_text_feature_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

SAMPLE_LAYER = "flow_run"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
TOKEN_SEPARATOR = " "

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

PUNCT_ONLY_PATTERN = re.compile(r"^[\W_]+$", re.UNICODE)


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


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def safe_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_int(value: object) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def build_event_id(question_id: str, user_id: str, time_seq: int) -> str:
    return f"{question_id}|{user_id}|{time_seq}"


def tokenize_text(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if jieba is not None:
        raw_tokens = jieba.lcut(cleaned, cut_all=False)
    else:  # pragma: no cover
        raw_tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", cleaned)

    tokens: list[str] = []
    for token in raw_tokens:
        token = clean_text(token)
        if not token:
            continue
        if PUNCT_ONLY_PATTERN.fullmatch(token):
            continue
        tokens.append(token)
    return tokens


def tokens_to_json(tokens: list[str]) -> str:
    return json.dumps(tokens, ensure_ascii=False)


def tokens_to_joined_text(tokens: list[str]) -> str:
    return TOKEN_SEPARATOR.join(tokens)


def prepare_flow_answers() -> pd.DataFrame:
    df = pd.read_parquet(ANSWER_SAMPLE_PATH).copy()
    df = df.loc[df["sample_layer"] == SAMPLE_LAYER].copy()
    df["question_id"] = df["question_id"].map(safe_str)
    df["user_id"] = df["user_id"].map(safe_str)
    df["answer"] = df["answer"].map(clean_text)
    df["question_desc"] = df["question_desc"].map(clean_text)
    df["time_seq"] = df["time_seq"].map(safe_int).astype("int64")
    df["seq"] = df["seq"].map(safe_int).astype("int64")
    df["answer_create_time"] = pd.to_datetime(df["answer_create_time"])
    df["event_id"] = [
        build_event_id(question_id, user_id, time_seq)
        for question_id, user_id, time_seq in zip(df["question_id"], df["user_id"], df["time_seq"])
    ]
    df = df.sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    return df


def prepare_flow_sentences(flow_event_ids: set[str], flow_question_ids: set[str]) -> pd.DataFrame:
    df = pd.read_parquet(SENTENCE_TABLE_PATH).copy()
    df["question_id"] = df["question_id"].map(safe_str)
    df["user_id"] = df["user_id"].map(safe_str)
    df["time_seq"] = df["time_seq"].map(safe_int).astype("int64")
    df["sentence_order"] = df["sentence_order"].map(safe_int).astype("int64")
    df["sentence_text"] = df["sentence_text"].map(clean_text)
    df["event_id"] = [
        build_event_id(question_id, user_id, time_seq)
        for question_id, user_id, time_seq in zip(df["question_id"], df["user_id"], df["time_seq"])
    ]
    df = df.loc[df["question_id"].isin(flow_question_ids) & df["event_id"].isin(flow_event_ids)].copy()
    df = df.sort_values(["question_id", "event_id", "sentence_order"], kind="stable").reset_index(drop=True)
    return df


def compute_prior_features(flow_answer_df: pd.DataFrame) -> pd.DataFrame:
    output_rows: list[dict[str, object]] = []

    for question_id, group_df in flow_answer_df.groupby("question_id", sort=False):
        group_df = group_df.sort_values(
            ["answer_create_time", "time_seq", "seq"],
            kind="stable",
        ).reset_index(drop=True)

        timestamp_groups: list[pd.DataFrame] = [
            timestamp_df.copy()
            for _, timestamp_df in group_df.groupby("answer_create_time", sort=False)
        ]

        prior_event_ids: list[str] = []
        prior_texts: list[str] = []
        prior_total_char_count = 0

        for timestamp_df in timestamp_groups:
            timestamp_df = timestamp_df.sort_values(["time_seq", "seq"], kind="stable").reset_index(drop=True)

            for row in timestamp_df.itertuples(index=False):
                current_answer_text = clean_text(row.answer)
                current_answer_char_count = len(current_answer_text)
                answer_tokens = tokenize_text(current_answer_text)

                output_rows.append(
                    {
                        "sample_layer": SAMPLE_LAYER,
                        "question_id": question_id,
                        "event_id": row.event_id,
                        "user_id": row.user_id,
                        "time_seq": int(row.time_seq),
                        "seq": int(row.seq),
                        "answer_create_time": pd.Timestamp(row.answer_create_time).isoformat(),
                        "question_desc": clean_text(row.question_desc),
                        "answer_text": current_answer_text,
                        "current_answer_char_count": current_answer_char_count,
                        "answer_tokens_json": tokens_to_json(answer_tokens),
                        "answer_tokens_joined": tokens_to_joined_text(answer_tokens),
                        "answer_token_count": len(answer_tokens),
                        "strict_prior_event_ids_json": json.dumps(prior_event_ids, ensure_ascii=False),
                        "strict_prior_answer_count_current": len(prior_event_ids),
                        "strict_prior_total_char_count_current": prior_total_char_count,
                        "strict_prior_text_concat": "\n\n".join(prior_texts),
                        "strict_prior_text_concat_char_count": sum(len(text) for text in prior_texts),
                    }
                )

            for row in timestamp_df.itertuples(index=False):
                answer_text = clean_text(row.answer)
                prior_event_ids.append(row.event_id)
                prior_texts.append(answer_text)
                prior_total_char_count += len(answer_text)

    output_df = pd.DataFrame.from_records(output_rows)
    output_df = output_df.sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    return output_df


def build_sentence_feature_table(flow_sentence_df: pd.DataFrame) -> pd.DataFrame:
    sentence_output_df = flow_sentence_df.copy()
    sentence_output_df["sentence_char_count"] = sentence_output_df["sentence_text"].str.len().astype("int64")

    sentence_tokens = sentence_output_df["sentence_text"].map(tokenize_text)
    sentence_output_df["sentence_tokens_json"] = sentence_tokens.map(tokens_to_json)
    sentence_output_df["sentence_tokens_joined"] = sentence_tokens.map(tokens_to_joined_text)
    sentence_output_df["sentence_token_count"] = sentence_tokens.map(len).astype("int64")

    return sentence_output_df[
        [
            "question_id",
            "event_id",
            "sentence_id",
            "sentence_order",
            "sentence_text",
            "sentence_char_count",
            "sentence_tokens_json",
            "sentence_tokens_joined",
            "sentence_token_count",
        ]
    ].copy()


def build_summary(answer_feature_df: pd.DataFrame, sentence_feature_df: pd.DataFrame) -> dict[str, object]:
    answer_rows = int(len(answer_feature_df))
    question_count = int(answer_feature_df["question_id"].nunique())
    sentence_rows = int(len(sentence_feature_df))
    answer_with_prior = int((answer_feature_df["strict_prior_answer_count_current"] > 0).sum())
    max_prior_answer_count = int(answer_feature_df["strict_prior_answer_count_current"].max()) if answer_rows else 0
    max_prior_char_count = int(
        answer_feature_df["strict_prior_total_char_count_current"].max()
    ) if answer_rows else 0

    return {
        "run_ts_utc": utc_now_iso(),
        "scope": "flow_run_text_features",
        "inputs": {
            "answer_sample_path": str(ANSWER_SAMPLE_PATH),
            "sentence_table_path": str(SENTENCE_TABLE_PATH),
            "sample_layer": SAMPLE_LAYER,
        },
        "outputs": {
            "answer_feature_path": str(ANSWER_OUTPUT_PATH),
            "sentence_feature_path": str(SENTENCE_OUTPUT_PATH),
        },
        "embedding_plan": {
            "model_name": EMBEDDING_MODEL_NAME,
            "library": "sentence_transformers",
            "recommended_device_from_user": "cuda",
            "current_env_cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
            "current_env_effective_device": "cuda"
            if (torch is not None and torch.cuda.is_available())
            else "cpu",
            "note": "本步先计算文本中间层特征；下一步可直接对 current_answer_text 与 strict_prior_text_concat 计算 BGE 向量。",
        },
        "tokenization": {
            "tool": "jieba.lcut" if jieba is not None else "regex_fallback",
            "drop_punctuation_only_tokens": True,
            "token_separator_for_joined_text": TOKEN_SEPARATOR,
        },
        "feature_definitions": {
            "strict_prior_answer_count_current": "同一 question_id 内，时间戳严格早于当前回答的前序回答数；同时间戳回答不互为前序。",
            "strict_prior_total_char_count_current": "严格前序回答文本的总字符数。",
            "current_answer_char_count": "当前回答文本字符数。",
            "strict_prior_text_concat": "严格前序回答按时间顺序拼接后的文本版本。",
        },
        "counts": {
            "flow_run_question_count": question_count,
            "flow_run_answer_rows": answer_rows,
            "flow_run_sentence_rows": sentence_rows,
            "answers_with_strict_prior": answer_with_prior,
            "answers_with_strict_prior_rate": round(answer_with_prior / answer_rows, 6) if answer_rows else 0.0,
            "max_strict_prior_answer_count": max_prior_answer_count,
            "max_strict_prior_total_char_count": max_prior_char_count,
        },
        "preview": {
            "answer_feature_head": answer_feature_df.head(5).to_dict(orient="records"),
            "sentence_feature_head": sentence_feature_df.head(5).to_dict(orient="records"),
        },
    }


def main() -> None:
    ensure_directories()

    flow_answer_df = prepare_flow_answers()
    flow_question_ids = set(flow_answer_df["question_id"].tolist())
    flow_event_ids = set(flow_answer_df["event_id"].tolist())

    answer_feature_df = compute_prior_features(flow_answer_df)
    sentence_feature_df = build_sentence_feature_table(
        prepare_flow_sentences(flow_event_ids=flow_event_ids, flow_question_ids=flow_question_ids)
    )

    answer_feature_df.to_parquet(ANSWER_OUTPUT_PATH, index=False)
    sentence_feature_df.to_parquet(SENTENCE_OUTPUT_PATH, index=False)

    summary = build_summary(answer_feature_df, sentence_feature_df)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05f_prepare_flow_run_text_features",
                "action": "compute_flow_run_answer_tokens_and_strict_prior_text_features",
                "status": "completed",
                "target_path": str(ANSWER_OUTPUT_PATH),
                "details": (
                    f"sample_layer={SAMPLE_LAYER}; answer_rows={len(answer_feature_df)}; "
                    f"questions={answer_feature_df['question_id'].nunique()}; model_plan={EMBEDDING_MODEL_NAME}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05f_prepare_flow_run_text_features",
                "action": "compute_flow_run_sentence_tokens",
                "status": "completed",
                "target_path": str(SENTENCE_OUTPUT_PATH),
                "details": (
                    f"sentence_rows={len(sentence_feature_df)}; "
                    f"cuda_available={summary['embedding_plan']['current_env_cuda_available']}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05f_prepare_flow_run_text_features",
                "action": "write_flow_run_text_feature_summary",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"answers_with_prior={summary['counts']['answers_with_strict_prior']}; "
                    f"max_prior_answers={summary['counts']['max_strict_prior_answer_count']}"
                ),
            },
        ]
    )

    print(f"Answer-level text features written to {ANSWER_OUTPUT_PATH}")
    print(f"Sentence-level text features written to {SENTENCE_OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
