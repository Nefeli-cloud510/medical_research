from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

INPUT_PATH = ANALYSIS_DIR / "major5_text_cleaning_prep_seq_clean.parquet"
OUTPUT_PATH = ANALYSIS_DIR / "major5_answer_sentences.parquet"
SUMMARY_PATH = AUDIT_DIR / "17_major5_sentence_split_summary.json"
CONDITION_SAMPLE_PATH = TABLES_DIR / "17_sentence_split_condition_event_samples_major5.csv"
DECIMAL_SAMPLE_PATH = TABLES_DIR / "17_sentence_split_decimal_event_samples_major5.csv"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

READ_COLUMNS = [
    "question_id",
    "user_id",
    "time_seq",
    "seq",
    "answer_create_time",
    "answer",
]

BATCH_SIZE = 50_000
SAMPLE_LIMIT = 200
CONDITION_KEYWORDS = ("如果", "若", "如", "出现", "必要时", "否则", "建议", "应", "需")
PUNCT_ONLY_PATTERN = re.compile(r"^[\s\W_]+$", re.UNICODE)
DECIMAL_PATTERN = re.compile(r"\d+\.\d+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
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


def answer_time_to_iso(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def is_sentence_boundary(text: str, idx: int) -> bool:
    char = text[idx]

    if char in "。！？；?!;":
        return True

    if char in ".．｡":
        prev_char = text[idx - 1] if idx > 0 else ""
        next_char = text[idx + 1] if idx + 1 < len(text) else ""
        if prev_char.isdigit() and next_char.isdigit():
            return False
        return True

    return False


def normalize_sentence(candidate: str) -> str:
    sentence = candidate.strip()
    if not sentence:
        return ""
    if PUNCT_ONLY_PATTERN.fullmatch(sentence):
        return ""
    return sentence


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    buffer: list[str] = []

    for idx, char in enumerate(text):
        if char == "\n":
            sentence = normalize_sentence("".join(buffer))
            if sentence:
                sentences.append(sentence)
            buffer = []
            continue

        buffer.append(char)
        if is_sentence_boundary(text, idx):
            sentence = normalize_sentence("".join(buffer))
            if sentence:
                sentences.append(sentence)
            buffer = []

    tail_sentence = normalize_sentence("".join(buffer))
    if tail_sentence:
        sentences.append(tail_sentence)

    return sentences


def build_event_id(question_id: str, user_id: str, time_seq: int) -> str:
    # Current major5 sample has no platform answer_id; build a stable, traceable event key instead.
    return f"{question_id}|{user_id}|{time_seq}"


def maybe_append_event_sample(
    sample_rows: list[dict[str, object]],
    limit: int,
    event_id: str,
    question_id: str,
    user_id: str,
    time_seq: int,
    seq: int,
    answer_create_time_iso: str,
    answer_text: str,
    sentences: list[str],
) -> None:
    if len(sample_rows) >= limit:
        return

    sample_rows.append(
        {
            "event_id": event_id,
            "question_id": question_id,
            "user_id": user_id,
            "time_seq": time_seq,
            "seq": seq,
            "answer_create_time": answer_create_time_iso,
            "sentence_count": len(sentences),
            "answer_text": answer_text,
            "split_preview": " || ".join(sentences),
        }
    )


def write_sample_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_directories()

    parquet_file = pq.ParquetFile(INPUT_PATH)
    writer: pq.ParquetWriter | None = None

    total_events = 0
    empty_answer_events = 0
    nonempty_answer_events = 0
    total_sentences = 0
    single_sentence_events = 0
    multi_sentence_events = 0
    condition_keyword_event_count = 0
    decimal_event_count = 0
    total_sentence_chars = 0
    max_sentence_char_count = 0
    min_sentence_char_count: int | None = None
    max_sentences_per_event = 0

    condition_samples: list[dict[str, object]] = []
    decimal_samples: list[dict[str, object]] = []

    try:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=READ_COLUMNS):
            batch_df = batch.to_pandas()
            output_rows: list[dict[str, object]] = []

            for row in batch_df.itertuples(index=False):
                total_events += 1

                question_id = safe_str(row.question_id)
                user_id = safe_str(row.user_id)
                time_seq = safe_int(row.time_seq)
                seq = safe_int(row.seq)
                answer_create_time_iso = answer_time_to_iso(row.answer_create_time)
                answer_text = clean_text(row.answer)

                if not answer_text:
                    empty_answer_events += 1
                    continue

                sentences = split_sentences(answer_text)
                if not sentences:
                    empty_answer_events += 1
                    continue

                nonempty_answer_events += 1
                total_sentences += len(sentences)
                max_sentences_per_event = max(max_sentences_per_event, len(sentences))
                if len(sentences) == 1:
                    single_sentence_events += 1
                else:
                    multi_sentence_events += 1

                has_condition_keyword = any(keyword in answer_text for keyword in CONDITION_KEYWORDS)
                has_decimal = bool(DECIMAL_PATTERN.search(answer_text))
                if has_condition_keyword:
                    condition_keyword_event_count += 1
                if has_decimal:
                    decimal_event_count += 1

                event_id = build_event_id(question_id, user_id, time_seq)

                if has_condition_keyword:
                    maybe_append_event_sample(
                        condition_samples,
                        SAMPLE_LIMIT,
                        event_id,
                        question_id,
                        user_id,
                        time_seq,
                        seq,
                        answer_create_time_iso,
                        answer_text,
                        sentences,
                    )
                if has_decimal:
                    maybe_append_event_sample(
                        decimal_samples,
                        SAMPLE_LIMIT,
                        event_id,
                        question_id,
                        user_id,
                        time_seq,
                        seq,
                        answer_create_time_iso,
                        answer_text,
                        sentences,
                    )

                for sentence_order, sentence_text in enumerate(sentences, start=1):
                    sentence_char_count = len(sentence_text)
                    total_sentence_chars += sentence_char_count
                    max_sentence_char_count = max(max_sentence_char_count, sentence_char_count)
                    if min_sentence_char_count is None:
                        min_sentence_char_count = sentence_char_count
                    else:
                        min_sentence_char_count = min(min_sentence_char_count, sentence_char_count)

                    output_rows.append(
                        {
                            "event_id": event_id,
                            "question_id": question_id,
                            "user_id": user_id,
                            "time_seq": time_seq,
                            "seq": seq,
                            "answer_create_time": answer_create_time_iso,
                            "sentence_id": f"{event_id}_s{sentence_order}",
                            "sentence_order": sentence_order,
                            "sentence_text": sentence_text,
                            "sentence_char_count": sentence_char_count,
                        }
                    )

            if output_rows:
                output_df = pd.DataFrame.from_records(output_rows)
                output_table = pa.Table.from_pandas(output_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(OUTPUT_PATH, output_table.schema, compression="snappy")
                writer.write_table(output_table)
    finally:
        if writer is not None:
            writer.close()

    write_sample_csv(CONDITION_SAMPLE_PATH, condition_samples)
    write_sample_csv(DECIMAL_SAMPLE_PATH, decimal_samples)

    average_sentences_per_nonempty_event = (
        round(total_sentences / nonempty_answer_events, 6) if nonempty_answer_events else 0.0
    )
    average_sentence_char_count = (
        round(total_sentence_chars / total_sentences, 6) if total_sentences else 0.0
    )

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "major5_answer_sentence_split",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "sentence_split_rules": {
            "primary_boundaries": ["。", "？", "！", "；", "newline"],
            "secondary_boundaries_supported": ["?", "!", ";", ".", "．", "｡"],
            "decimal_protection": "数字内部的小数点不作为句界。",
            "condition_clause_protection_note": "不以逗号为句界，尽量保留“如果…，应…”等条件句在同一语义单元中。",
        },
        "event_id_definition": {
            "field_name": "event_id",
            "construction": "question_id|user_id|time_seq",
            "note": "当前 major5 主样本无平台真实 answer_id，因此 event_id 为可回溯代理事件键，不是平台真实回答 ID。",
        },
        "event_stats": {
            "total_events_scanned": total_events,
            "nonempty_answer_events": nonempty_answer_events,
            "empty_answer_events": empty_answer_events,
            "single_sentence_events": single_sentence_events,
            "multi_sentence_events": multi_sentence_events,
            "max_sentences_per_event": max_sentences_per_event,
            "condition_keyword_event_count": condition_keyword_event_count,
            "decimal_event_count": decimal_event_count,
        },
        "sentence_stats": {
            "total_sentences": total_sentences,
            "average_sentences_per_nonempty_event": average_sentences_per_nonempty_event,
            "average_sentence_char_count": average_sentence_char_count,
            "min_sentence_char_count": min_sentence_char_count or 0,
            "max_sentence_char_count": max_sentence_char_count,
        },
        "output_fields": [
            "event_id",
            "question_id",
            "user_id",
            "time_seq",
            "seq",
            "answer_create_time",
            "sentence_id",
            "sentence_order",
            "sentence_text",
            "sentence_char_count",
        ],
        "manual_review_outputs": {
            "condition_event_samples": str(CONDITION_SAMPLE_PATH),
            "decimal_event_samples": str(DECIMAL_SAMPLE_PATH),
            "sample_limit_each": SAMPLE_LIMIT,
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05d_split_major5_answer_sentences",
                "action": "split_answer_text_into_sentences_and_write_sentence_table",
                "status": "completed",
                "target_path": str(OUTPUT_PATH),
                "details": (
                    f"input={INPUT_PATH.name}; total_events={total_events}; "
                    f"nonempty_answer_events={nonempty_answer_events}; total_sentences={total_sentences}; "
                    f"event_id_proxy=question_id|user_id|time_seq"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05d_split_major5_answer_sentences",
                "action": "write_sentence_split_summary",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"avg_sentences_per_event={average_sentences_per_nonempty_event}; "
                    f"condition_events={condition_keyword_event_count}; decimal_events={decimal_event_count}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05d_split_major5_answer_sentences",
                "action": "write_manual_review_samples",
                "status": "completed",
                "target_path": str(CONDITION_SAMPLE_PATH),
                "details": (
                    f"condition_samples={len(condition_samples)}; decimal_samples={len(decimal_samples)}; "
                    f"decimal_sample_path={DECIMAL_SAMPLE_PATH.name}"
                ),
            },
        ]
    )

    print(f"Sentence table written to {OUTPUT_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")
    print(f"Condition samples written to {CONDITION_SAMPLE_PATH}")
    print(f"Decimal samples written to {DECIMAL_SAMPLE_PATH}")


if __name__ == "__main__":
    main()
