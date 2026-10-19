from __future__ import annotations

import csv
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

ANSWER_INPUT_PATH = ANALYSIS_DIR / "flow_run100_answer_text_features.parquet"
SENTENCE_INPUT_PATH = ANALYSIS_DIR / "flow_run_sentence_text_features.parquet"

SENTENCE_OUTPUT_PATH = ANALYSIS_DIR / "flow_run100_sentence_text_for_topic_clean.csv"
ANSWER_OUTPUT_PATH = ANALYSIS_DIR / "flow_run100_answer_text_for_topic_clean.csv"
SUMMARY_PATH = AUDIT_DIR / "20j_flow_run100_topic_text_clean_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]


GREETING_ONLY_RE = re.compile(
    r"^\s*(你好|您好|你好啊|您好啊|很高兴为您解答|很高兴为你解答|感谢咨询|谢谢咨询|祝您健康|祝您好运|祝好|祝您生活愉快)\s*[!！。．…]*\s*$"
)
PURE_NUMBERING_RE = re.compile(r"^\s*[(（]?\s*[0-9一二三四五六七八九十]+\s*[)）]?\s*[\.。、:：]?\s*$")
LIST_PREFIX_RE = re.compile(
    r"^\s*((第\s*[0-9一二三四五六七八九十]+\s*[点条项]\s*[:：]?)|"
    r"([(（]\s*[0-9一二三四五六七八九十]+\s*[)）]\s*[\.。、:：]?)|"
    r"([0-9一二三四五六七八九十]+\s*[\.、:：]))\s*"
)
PREFIX_WORD_RE = re.compile(r"^\s*(建议|治疗建议|指导意见|初步意见|初步指导意见|意见|分析|答复|回复)\s*[:：]\s*")
PURE_SYMBOL_RE = re.compile(r"^[\W_]+$", flags=re.UNICODE)
MULTISPACE_RE = re.compile(r"\s+")


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


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def clean_sentence_for_topic(text: str) -> tuple[str, bool, str]:
    original = normalize_text(text)
    if not original:
        return "", True, "empty_after_normalize"

    cleaned = PREFIX_WORD_RE.sub("", original)
    cleaned = LIST_PREFIX_RE.sub("", cleaned)
    cleaned = normalize_text(cleaned)

    if not cleaned:
        return "", True, "empty_after_prefix_strip"
    if PURE_NUMBERING_RE.fullmatch(cleaned):
        return "", True, "pure_numbering"
    if GREETING_ONLY_RE.fullmatch(cleaned):
        return "", True, "greeting_only"
    if PURE_SYMBOL_RE.fullmatch(cleaned):
        return "", True, "pure_symbol"
    if len(cleaned) <= 2:
        return "", True, "too_short"

    return cleaned, False, ""


def build_answer_level_clean_text(sentence_df: pd.DataFrame) -> pd.DataFrame:
    answer_records = []
    for event_id, group_df in sentence_df.groupby("event_id", sort=False):
        group_df = group_df.sort_values("sentence_order", kind="stable")
        kept_df = group_df.loc[~group_df["drop_for_topic_cleaning"]].copy()
        cleaned_sentences = kept_df["sentence_text_topic_clean"].tolist()
        answer_text_topic_clean = " ".join(cleaned_sentences).strip()

        base_row = group_df.iloc[0]
        answer_records.append(
            {
                "sample_layer": base_row["sample_layer"],
                "question_id": base_row["question_id"],
                "event_id": event_id,
                "user_id": base_row["user_id"],
                "time_seq": int(base_row["time_seq"]),
                "seq": int(base_row["seq"]),
                "answer_create_time": str(base_row["answer_create_time"]),
                "question_text": base_row["question_desc"],
                "focal_answer_text_raw": base_row["answer_text"],
                "answer_text_topic_clean": answer_text_topic_clean if answer_text_topic_clean else None,
                "kept_sentence_count": int((~group_df["drop_for_topic_cleaning"]).sum()),
                "dropped_sentence_count": int(group_df["drop_for_topic_cleaning"].sum()),
                "has_clean_topic_text": bool(answer_text_topic_clean),
            }
        )

    answer_df = pd.DataFrame.from_records(answer_records)
    answer_df = answer_df.sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    return answer_df


def build_summary(sentence_df: pd.DataFrame, answer_df: pd.DataFrame) -> dict[str, object]:
    dropped = sentence_df["drop_for_topic_cleaning"]
    preview_sentence_df = sentence_df.head(15).copy()
    preview_answer_df = answer_df.head(10).copy()
    if "answer_create_time" in preview_sentence_df.columns:
        preview_sentence_df["answer_create_time"] = preview_sentence_df["answer_create_time"].astype(str)
    if "answer_create_time" in preview_answer_df.columns:
        preview_answer_df["answer_create_time"] = preview_answer_df["answer_create_time"].astype(str)
    return {
        "run_ts_utc": utc_now_iso(),
        "scope": "flow_run100_topic_text_cleaning_for_sentence_bertopic",
        "inputs": {
            "answer_input_path": str(ANSWER_INPUT_PATH),
            "sentence_input_path": str(SENTENCE_INPUT_PATH),
        },
        "outputs": {
            "sentence_output_path": str(SENTENCE_OUTPUT_PATH),
            "answer_output_path": str(ANSWER_OUTPUT_PATH),
        },
        "rules": [
            "解码 HTML 实体并规范空白字符",
            "删除句首的编号前缀与“建议: / 指导意见:”等格式前缀",
            "删除纯编号句、纯礼貌句、纯符号句、清洗后长度<=2的句子",
            "保留有实质医学内容的短句，不做医学术语改写",
        ],
        "counts": {
            "sentence_rows_total": int(len(sentence_df)),
            "sentence_rows_kept": int((~dropped).sum()),
            "sentence_rows_dropped": int(dropped.sum()),
            "sentence_keep_rate": round(float((~dropped).mean()), 6),
            "answer_rows_total": int(len(answer_df)),
            "answers_with_clean_topic_text": int(answer_df["has_clean_topic_text"].sum()),
            "answers_without_clean_topic_text": int((~answer_df["has_clean_topic_text"]).sum()),
        },
        "drop_reason_distribution": (
            sentence_df.loc[dropped, "topic_clean_drop_reason"]
            .value_counts(dropna=False)
            .rename_axis("drop_reason")
            .reset_index(name="sentence_count")
            .to_dict(orient="records")
        ),
        "preview_sentences": preview_sentence_df.to_dict(orient="records"),
        "preview_answers": preview_answer_df.to_dict(orient="records"),
    }


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    answer_df = pd.read_parquet(ANSWER_INPUT_PATH).copy()
    answer_df["question_id"] = answer_df["question_id"].astype(str)
    answer_df["answer_create_time"] = pd.to_datetime(answer_df["answer_create_time"])

    sentence_df = pd.read_parquet(SENTENCE_INPUT_PATH).copy()
    sentence_df["question_id"] = sentence_df["question_id"].astype(str)

    flow_run100_question_ids = set(answer_df["question_id"].tolist())
    sentence_df = sentence_df.loc[sentence_df["question_id"].isin(flow_run100_question_ids)].copy()
    sentence_df = sentence_df.merge(
        answer_df[
            [
                "event_id",
                "sample_layer",
                "user_id",
                "time_seq",
                "seq",
                "answer_create_time",
                "question_desc",
                "answer_text",
            ]
        ],
        on="event_id",
        how="left",
        validate="many_to_one",
    )

    needed_cols = [
        "sample_layer",
        "question_id",
        "event_id",
        "user_id",
        "time_seq",
        "seq",
        "answer_create_time",
        "question_desc",
        "answer_text",
        "sentence_id",
        "sentence_order",
        "sentence_text",
        "sentence_char_count",
    ]
    sentence_df = sentence_df[needed_cols].sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq", "sentence_order"],
        kind="stable",
    ).reset_index(drop=True)

    clean_results = sentence_df["sentence_text"].fillna("").astype(str).apply(clean_sentence_for_topic)
    sentence_df["sentence_text_topic_clean"] = clean_results.map(lambda x: x[0])
    sentence_df["drop_for_topic_cleaning"] = clean_results.map(lambda x: x[1])
    sentence_df["topic_clean_drop_reason"] = clean_results.map(lambda x: x[2])

    answer_clean_df = build_answer_level_clean_text(sentence_df)

    sentence_save_df = sentence_df.copy()
    sentence_save_df["answer_create_time"] = sentence_save_df["answer_create_time"].astype(str)
    sentence_save_df.to_csv(SENTENCE_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    answer_clean_df.to_csv(ANSWER_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    summary = build_summary(sentence_df=sentence_df, answer_df=answer_clean_df)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05o_clean_flow_run100_topic_text",
                "action": "write_sentence_level_topic_clean_text",
                "status": "completed",
                "target_path": str(SENTENCE_OUTPUT_PATH),
                "details": (
                    f"sentence_rows_total={summary['counts']['sentence_rows_total']}; "
                    f"sentence_rows_kept={summary['counts']['sentence_rows_kept']}; "
                    f"sentence_rows_dropped={summary['counts']['sentence_rows_dropped']}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05o_clean_flow_run100_topic_text",
                "action": "write_answer_level_topic_clean_text",
                "status": "completed",
                "target_path": str(ANSWER_OUTPUT_PATH),
                "details": (
                    f"answer_rows_total={summary['counts']['answer_rows_total']}; "
                    f"answers_with_clean_topic_text={summary['counts']['answers_with_clean_topic_text']}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05o_clean_flow_run100_topic_text",
                "action": "write_topic_clean_summary",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": "topic modeling sentence cleaning completed",
            },
        ]
    )

    print(f"Sentence-level cleaned topic text written to {SENTENCE_OUTPUT_PATH}")
    print(f"Answer-level cleaned topic text written to {ANSWER_OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
