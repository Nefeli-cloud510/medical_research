from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
RUN_ROOT = ROOT_DIR / "outputs" / "topic_ab_run_20261020"

ANSWER_CSV = ANALYSIS_DIR / "flow_run100_answer_text_for_topic_clean.csv"
SENTENCE_CSV = ANALYSIS_DIR / "flow_run100_sentence_text_for_topic_clean.csv"
LOCAL_MODEL_DIR = ROOT_DIR / "models" / "bge-base-zh-v1.5"
MODEL_NAME = "BAAI/bge-base-zh-v1.5"

OUTLIER_TOPIC_ID = -1
BATCH_SIZE = 32
ID_STR_COLUMNS = ["question_id", "event_id", "user_id"]

UNIT_CONFIG = {
    "sentence": {
        "dir_name": "A_sentence_no_reduce",
        "display_name": "句子级 A",
        "csv": SENTENCE_CSV,
        "text_col": "sentence_text_topic_clean",
        "doc_id_col": "sentence_id",
        "sort_cols": ["question_id", "time_seq", "answer_create_time", "seq", "sentence_order"],
    },
    "answer": {
        "dir_name": "B_answer_no_reduce",
        "display_name": "回答级 B",
        "csv": ANSWER_CSV,
        "text_col": "answer_text_topic_clean",
        "doc_id_col": "event_id",
        "sort_cols": ["question_id", "time_seq", "answer_create_time", "seq"],
    },
}

TEMPLATE_RE = re.compile(
    r"(您好|你好|很高兴|感谢咨询|谢谢咨询|祝您|祝你|早日康复|生活愉快|身体健康|"
    r"希望.*帮助|仅供参考|如有疑问|随时.*咨询|不适随诊|希望对您)"
)
BOILERPLATE_RE = re.compile(
    r"(建议您|建议到|定期复查|及时就医|进一步检查|必要时|以上.*建议|到.*就诊)"
)
MEDICAL_MARKER_RE = re.compile(
    r"(病|药|检查|治疗|手术|症状|诊断|阳性|阴性|风险|剂量|医院|医生|血糖|血压|肿瘤|感染|复查)"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _load_05p_module():
    cache_attr = "_topic_pipeline_05p_module"
    if hasattr(_load_05p_module, cache_attr):
        return getattr(_load_05p_module, cache_attr)
    spec = importlib.util.spec_from_file_location(
        "topic_pipeline_05p",
        SCRIPTS_DIR / "05p_bertopic.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setattr(_load_05p_module, cache_attr, module)
    return module


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in (
        "bertopic",
        "sentence_transformers",
        "sklearn",
        "jieba",
        "numpy",
        "pandas",
        "umap",
        "hdbscan",
    ):
        try:
            module = __import__(package_name)
            versions[package_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package_name] = "import_error"
    return versions


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dtype_map = {column: str for column in ID_STR_COLUMNS}
    answer_df = pd.read_csv(ANSWER_CSV, encoding="utf-8-sig", dtype=dtype_map)
    answer_df["answer_create_time"] = pd.to_datetime(answer_df["answer_create_time"])
    answer_df["time_seq"] = pd.to_numeric(answer_df["time_seq"], errors="coerce").astype("Int64")
    answer_df["seq"] = pd.to_numeric(answer_df["seq"], errors="coerce").astype("Int64")
    if "has_clean_topic_text" in answer_df.columns:
        answer_df = answer_df.loc[_load_05p_module().coerce_bool_series(answer_df["has_clean_topic_text"])].copy()
    answer_df = answer_df.sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)

    question_ids = set(answer_df["question_id"].astype(str).tolist())
    sentence_df = pd.read_csv(SENTENCE_CSV, encoding="utf-8-sig", dtype=dtype_map)
    sentence_df["answer_create_time"] = pd.to_datetime(sentence_df["answer_create_time"])
    sentence_df["time_seq"] = pd.to_numeric(sentence_df["time_seq"], errors="coerce").astype("Int64")
    sentence_df["seq"] = pd.to_numeric(sentence_df["seq"], errors="coerce").astype("Int64")
    sentence_df["question_id"] = sentence_df["question_id"].astype(str)
    sentence_df = sentence_df.loc[sentence_df["question_id"].isin(question_ids)].copy()
    if "drop_for_topic_cleaning" in sentence_df.columns:
        sentence_df = sentence_df.loc[
            ~_load_05p_module().coerce_bool_series(sentence_df["drop_for_topic_cleaning"])
        ].copy()
    sentence_df["sentence_text_topic_clean"] = sentence_df["sentence_text_topic_clean"].fillna("").astype(str)
    sentence_df = sentence_df.loc[sentence_df["sentence_text_topic_clean"].str.strip() != ""].copy()
    sentence_df = sentence_df.sort_values(
        ["question_id", "time_seq", "answer_create_time", "seq", "sentence_order"],
        kind="stable",
    ).reset_index(drop=True)
    return answer_df, sentence_df


def text_payload_hash(df: pd.DataFrame, text_col: str) -> str:
    rows = []
    for row in df.itertuples(index=False):
        parts = [
            str(row.question_id),
            str(row.event_id),
            str(getattr(row, text_col)),
        ]
        rows.append("\u001f".join(parts))
    payload = "\u001e".join(rows)
    return sha256_text(payload)


def prepare_chunks(
    docs: list[str],
    tokenizer,
    max_seq_length: int,
) -> tuple[list[str], list[tuple[int, int]], int, int, int]:
    capacity = max_seq_length - 2
    flat_chunks: list[str] = []
    doc_chunk_ranges: list[tuple[int, int]] = []
    over_capacity = 0
    total_chunks = 0
    max_chunks_per_doc = 0
    for text in docs:
        input_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if len(input_ids) <= capacity:
            chunks = [text]
        else:
            chunks = []
            for start in range(0, len(input_ids), capacity):
                chunk_text = tokenizer.decode(
                    input_ids[start : start + capacity],
                    skip_special_tokens=True,
                ).strip()
                if chunk_text:
                    chunks.append(chunk_text)
            over_capacity += 1
        start = len(flat_chunks)
        flat_chunks.extend(chunks)
        doc_chunk_ranges.append((start, len(chunks)))
        total_chunks += len(chunks)
        max_chunks_per_doc = max(max_chunks_per_doc, len(chunks))
    return flat_chunks, doc_chunk_ranges, over_capacity, total_chunks, max_chunks_per_doc


def encode_docs(
    model: SentenceTransformer,
    docs: list[str],
    max_seq_length: int,
) -> tuple[np.ndarray, dict[str, object]]:
    tokenizer = model.tokenizer
    flat_chunks, doc_chunk_ranges, over_capacity, total_chunks, max_chunks_per_doc = prepare_chunks(
        docs, tokenizer, max_seq_length
    )
    if not flat_chunks:
        raise ValueError("No text chunks to encode.")
    flat_embeddings = model.encode(
        flat_chunks,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    embeddings = np.zeros((len(docs), flat_embeddings.shape[1]), dtype=np.float32)
    for index, (start, chunk_count) in enumerate(doc_chunk_ranges):
        if chunk_count == 1:
            embeddings[index] = flat_embeddings[start]
            continue
        chunk_vectors = flat_embeddings[start : start + chunk_count]
        mean_vector = chunk_vectors.mean(axis=0)
        norm = float(np.linalg.norm(mean_vector))
        embeddings[index] = mean_vector / norm if norm > 0 else mean_vector
    stats = {
        "max_seq_length": max_seq_length,
        "chunk_capacity": max_seq_length - 2,
        "special_tokens_reserved": 2,
        "chunking_strategy": "token_window_then_decode",
        "docs_over_capacity": over_capacity,
        "total_flat_chunks": total_chunks,
        "max_chunks_per_doc": max_chunks_per_doc,
        "normalize_embeddings": True,
        "batch_size": BATCH_SIZE,
    }
    return embeddings, stats


def load_or_build_embeddings(
    unit: str,
    docs: list[str],
    model: SentenceTransformer,
    model_source: str,
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, object], bool]:
    max_seq_length = int(model.get_max_seq_length())
    text_hashes = [sha256_text(text) for text in docs]
    input_hash = sha256_text("\u001f".join(text_hashes))
    cache_key = sha256_text(
        json.dumps(
            {
                "input_hash": input_hash,
                "model_source": model_source,
                "max_seq_length": max_seq_length,
                "chunk_capacity": max_seq_length - 2,
            },
            sort_keys=True,
        )
    )
    npy_path = cache_dir / f"{unit}_embeddings.npy"
    manifest_path = cache_dir / f"{unit}_manifest.json"
    hashes_path = cache_dir / f"{unit}_text_hashes.json"
    if npy_path.exists() and manifest_path.exists() and hashes_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            hashes_manifest = json.loads(hashes_path.read_text(encoding="utf-8"))
            if (
                manifest.get("cache_key") == cache_key
                and manifest.get("rows") == len(docs)
                and hashes_manifest.get("input_hash") == input_hash
                and hashes_manifest.get("hashes") == text_hashes
            ):
                cached = np.load(npy_path)
                if cached.shape[0] == len(docs):
                    return cached.astype(np.float32, copy=False), manifest, True
        except Exception:
            pass

    embeddings, chunk_stats = encode_docs(model, docs, max_seq_length)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, embeddings)
    hashes_manifest = {
        "unit": unit,
        "input_hash": input_hash,
        "algorithm": "sha256 of text in stable document order",
        "hashes": text_hashes,
    }
    hashes_path.write_text(json.dumps(hashes_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "unit": unit,
        "rows": len(docs),
        "input_hash": input_hash,
        "cache_key": cache_key,
        "model_name": MODEL_NAME,
        "model_source_used": model_source,
        "created_utc": utc_now_iso(),
        "embedding_columns": int(embeddings.shape[1]),
        "text_hash_file": str(hashes_path),
        "chunking": chunk_stats,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return embeddings, manifest, False


def save_topic_model(topic_model, unit_dir: Path, docs: list[str], topics: list[int]) -> dict[str, object]:
    save_result: dict[str, object] = {}
    try:
        model_dir = unit_dir / "bertopic_model"
        topic_model.save(
            str(model_dir),
            serialization="safetensors",
            save_embedding_model=False,
        )
        save_result["saved"] = True
        save_result["path"] = str(model_dir)
    except Exception as exc:  # pragma: no cover
        save_result["saved"] = False
        save_result["error"] = f"{type(exc).__name__}: {exc}"

    topic_labels = [int(topic) for topic in topics]
    topics_path = unit_dir / "topics_np.json"
    topics_path.write_text(
        json.dumps({"docs_count": len(topic_labels), "topics": topic_labels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return save_result


def unit_df(unit: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    answer_df, sentence_df = read_inputs()
    config = UNIT_CONFIG[unit]
    if unit == "sentence":
        docs_df = sentence_df.copy()
    else:
        docs_df = answer_df.copy()
    docs_df = docs_df.sort_values(config["sort_cols"], kind="stable").reset_index(drop=True)
    docs_df[config["text_col"]] = docs_df[config["text_col"]].fillna("").astype(str)
    return answer_df, sentence_df, docs_df


def fit_unit(unit: str) -> None:
    module = _load_05p_module()
    unit_dir = RUN_ROOT / UNIT_CONFIG[unit]["dir_name"]
    unit_dir.mkdir(parents=True, exist_ok=True)
    answer_df, sentence_df, docs_df = unit_df(unit)
    config = UNIT_CONFIG[unit]
    text_col = config["text_col"]
    docs = docs_df[text_col].tolist()
    print(f"[{unit}] loading embedding model")
    model, device, model_source = module.load_embedding_model()
    cache_dir = unit_dir / "assets" / "embeddings"
    embeddings, embedding_manifest, cache_hit = load_or_build_embeddings(
        unit=unit,
        docs=docs,
        model=model,
        model_source=model_source,
        cache_dir=cache_dir,
    )
    print(
        f"[{unit}] embeddings rows={embeddings.shape[0]} cols={embeddings.shape[1]} "
        f"device={device} cache_hit={cache_hit}"
    )

    topic_model = module.build_topic_model()
    print(f"[{unit}] starting BERTopic fit_transform without nr_topics compression")
    topics, _ = topic_model.fit_transform(docs, embeddings)
    topics = [int(topic) for topic in topics]
    labels_df = docs_df.copy()
    labels_df["topic_id"] = topics
    labels_df.to_csv(unit_dir / f"{unit}_labels.csv", index=False, encoding="utf-8-sig")

    topic_info_df = topic_model.get_topic_info().copy()
    topic_info_records = [
        {
            key: (value if isinstance(value, (int, float, str, list, type(None))) else str(value))
            for key, value in row.items()
        }
        for row in topic_info_df.to_dict(orient="records")
    ]
    (unit_dir / "topic_info.json").write_text(
        json.dumps(topic_info_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_result = save_topic_model(topic_model, unit_dir, docs, topics)

    counts = Counter(topics)
    valid_topic_ids = sorted({topic for topic in counts if topic != OUTLIER_TOPIC_ID})
    noise_count = int(counts.get(OUTLIER_TOPIC_ID, 0))
    total_rows = len(topics)
    max_topic, max_count = max(counts.items(), key=lambda item: item[1])
    if unit == "sentence":
        valid_mask = labels_df["topic_id"] != OUTLIER_TOPIC_ID
        unique_answer_events = labels_df.loc[valid_mask, "event_id"].astype(str).unique()
        answer_coverage = float(len(set(unique_answer_events)) / len(answer_df)) if len(answer_df) else 0.0
    else:
        answer_coverage = float(1.0 - noise_count / total_rows)

    umap_params = json_safe(topic_model.umap_model.get_params())
    hdbscan_params = json_safe(topic_model.hdbscan_model.get_params())
    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": f"flow_run100_topic_ab_{unit}_no_reduce",
        "unit_display_name": UNIT_CONFIG[unit]["display_name"],
        "inputs": {
            "csv": str(UNIT_CONFIG[unit]["csv"]),
            "text_column": text_col,
            "rows": total_rows,
            "payload_hash": text_payload_hash(docs_df, text_col),
        },
        "outputs": {
            "run_dir": str(unit_dir),
            "labels_csv": str(unit_dir / f"{unit}_labels.csv"),
            "topic_info_json": str(unit_dir / "topic_info.json"),
            "embeddings_npy": str(cache_dir / f"{unit}_embeddings.npy"),
        },
        "algorithm": {
            "topic_model": "BERTopic",
            "language": "multilingual",
            "embedding_model": MODEL_NAME,
            "embedding_model_source_used": model_source,
            "device_used": device,
            "vectorizer_min_df": 1,
            "vectorizer_ngram_range": [1, 2],
            "requested_nr_topics_including_noise": None,
            "topic_reduction_applied": False,
            "actual_valid_topic_k": len(valid_topic_ids),
            "initial_topic_count_including_noise": len(counts),
            "umap_params": umap_params,
            "hdbscan_params": hdbscan_params,
            "embedding_cache_hit": cache_hit,
            "embedding_manifest": embedding_manifest,
        },
        "counts": {
            "rows_total": total_rows,
            "valid_topic_rows": total_rows - noise_count,
            "noise_rows": noise_count,
            "noise_rate": round(float(noise_count / total_rows), 6),
            "valid_topic_count": len(valid_topic_ids),
            "topic_counts": {str(topic): int(count) for topic, count in sorted(counts.items())},
            "max_topic_id": int(max_topic),
            "max_topic_count": int(max_count),
            "max_topic_share": round(float(max_count / total_rows), 6),
            "answer_level_coverage": round(float(answer_coverage), 6),
        },
        "model_save": save_result,
        "script": {
            "canonical": "scripts/05r_topic_ab_pipeline.py",
            "sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        },
        "versions": dependency_versions(),
        "notes": [
            "No nr_topics reduction was requested; these are HDBSCAN initial topics.",
            "Topic id -1 is treated as noise and is not a valid knowledge topic.",
        ],
    }
    (unit_dir / "fit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[{unit}] rows={total_rows} topics={len(counts)} valid={len(valid_topic_ids)} "
        f"noise={noise_count} ({noise_count / total_rows:.1%}) max_topic={max_topic}:{max_count}"
    )


def normalize_info_list(records: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    def as_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if value is None:
            return []
        return [str(value)]
    if "Representation" in df.columns:
        df["Representation"] = df["Representation"].apply(as_list)
    if "Representative_Docs" in df.columns:
        df["Representative_Docs"] = df["Representative_Docs"].apply(as_list)
    return df


def export_unit(unit: str) -> None:
    unit_dir = RUN_ROOT / UNIT_CONFIG[unit]["dir_name"]
    labels_path = unit_dir / f"{unit}_labels.csv"
    info_path = unit_dir / "topic_info.json"
    if not labels_path.exists() or not info_path.exists():
        raise FileNotFoundError(f"Missing fitted outputs for unit {unit} in {unit_dir}")

    answer_df, sentence_df, docs_df = unit_df(unit)
    labels_df = pd.read_csv(labels_path, dtype={column: str for column in ID_STR_COLUMNS})
    if "topic_id" in labels_df.columns:
        labels_df["topic_id"] = labels_df["topic_id"].astype(int)
    info_records = json.loads(info_path.read_text(encoding="utf-8"))
    info_df = normalize_info_list(info_records)
    topic_to_info = {
        int(row.Topic): row._asdict()
        for row in info_df.itertuples(index=False)
    }

    audit_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    valid_topic_ids = sorted({int(value) for value in labels_df["topic_id"].unique() if int(value) != OUTLIER_TOPIC_ID})
    rng = np.random.default_rng(20261020)
    for topic_id in valid_topic_ids:
        topic_labels = labels_df.loc[labels_df["topic_id"] == topic_id].copy()
        count = len(topic_labels)
        question_count = int(topic_labels["question_id"].nunique())
        keywords = list(topic_to_info.get(topic_id, {}).get("Representation") or [])[:20]
        rep_docs = list(topic_to_info.get(topic_id, {}).get("Representative_Docs") or [])[:5]
        if len(rep_docs) < 3:
            extra = [
                text
                for text in topic_labels["answer_text_topic_clean"]
                if unit == "answer" and text not in set(rep_docs)
            ]
            if unit == "sentence":
                extra = [
                    text
                    for text in topic_labels["sentence_text_topic_clean"]
                    if text not in set(rep_docs)
                ]
            rep_docs = (rep_docs + extra)[:5]
        random_sample = topic_labels.sample(n=min(10, count), random_state=20261020, replace=False)
        if unit == "sentence":
            random_texts = random_sample["sentence_text_topic_clean"].tolist()
        else:
            random_texts = random_sample["answer_text_topic_clean"].tolist()

        template_share = 0.0
        marker_share = 0.0
        if count:
            if unit == "sentence":
                topic_texts = topic_labels["sentence_text_topic_clean"].fillna("").astype(str).tolist()
            else:
                topic_texts = topic_labels["answer_text_topic_clean"].fillna("").astype(str).tolist()
            template_share = float(np.mean([bool(TEMPLATE_RE.search(text) or BOILERPLATE_RE.search(text)) for text in topic_texts]))
            marker_share = float(np.mean([bool(MEDICAL_MARKER_RE.search(text)) for text in topic_texts]))

        audit_rows.append(
            {
                "topic_id": topic_id,
                "topic_doc_count": count,
                "involved_question_count": question_count,
                "keywords_zh_top20": "、".join(keywords),
                "representative_docs_zh": "\n---\n".join(rep_docs[:5]),
                "random_docs_zh": "\n---\n".join(random_texts[:10]),
                "template_share_heuristic": round(template_share, 4),
                "medical_marker_share_heuristic": round(marker_share, 4),
                "template_dominant_heuristic": bool(template_share >= 0.6),
                "candidate_name_zh": "",
                "one_line_definition": "",
                "mixed_medical_content_notes": "",
                "manual_flag": "",
            }
        )
        for text in rep_docs:
            sample_rows.append(
                {
                    "unit": unit,
                    "topic_id": topic_id,
                    "sample_type": "representative",
                    "question_id": "",
                    "doc_id": "",
                    "text": text,
                }
            )
        for _, row in random_sample.iterrows():
            sample_rows.append(
                {
                    "unit": unit,
                    "topic_id": topic_id,
                    "sample_type": "random",
                    "question_id": str(row["question_id"]),
                    "doc_id": str(row[UNIT_CONFIG[unit]["doc_id_col"]]),
                    "text": str(row[UNIT_CONFIG[unit]["text_col"]]),
                }
            )

    audit_df = pd.DataFrame.from_records(audit_rows)
    audit_df.to_csv(unit_dir / f"{unit}_topic_audit.csv", index=False, encoding="utf-8-sig")
    sample_df = pd.DataFrame.from_records(sample_rows)
    sample_df.to_csv(unit_dir / f"{unit}_text_samples.csv", index=False, encoding="utf-8-sig")

    labels_df["time_seq"] = pd.to_numeric(labels_df["time_seq"], errors="coerce").astype("Int64")
    labels_df["seq"] = pd.to_numeric(labels_df["seq"], errors="coerce").astype("Int64")
    if unit == "sentence":
        stats_rows: list[dict[str, object]] = []
        valid_topic_set = set(valid_topic_ids)
        for row in answer_df.itertuples(index=False):
            slice_df = labels_df.loc[labels_df["event_id"] == row.event_id].copy()
            total_sentences = len(slice_df)
            valid_slice = slice_df.loc[slice_df["topic_id"].isin(valid_topic_set)]
            valid_count = len(valid_slice)
            noise_count = int((slice_df["topic_id"] == OUTLIER_TOPIC_ID).sum())
            dominant_topic = None
            dominant_share = float("nan")
            topic_coverage = 0
            if valid_count:
                value_counts = valid_slice["topic_id"].value_counts()
                dominant_topic = int(value_counts.idxmax())
                dominant_share = float(value_counts.max() / valid_count)
                topic_coverage = int(value_counts.nunique())
            stats_rows.append(
                {
                    "question_id": str(row.question_id),
                    "event_id": str(row.event_id),
                    "user_id": str(row.user_id),
                    "time_seq": int(row.time_seq),
                    "seq": int(row.seq),
                    "answer_create_time": pd.Timestamp(row.answer_create_time).isoformat(),
                    "total_sentence_count": int(total_sentences),
                    "valid_topic_sentence_count": int(valid_count),
                    "noise_sentence_count": noise_count,
                    "noise_sentence_share": round(float(noise_count / total_sentences), 6) if total_sentences else None,
                    "topic_coverage": topic_coverage,
                    "dominant_topic_id": dominant_topic,
                    "dominant_topic_share": round(dominant_share, 6) if pd.notna(dominant_share) else None,
                    "has_valid_topic": bool(valid_count > 0),
                }
            )
        pd.DataFrame.from_records(stats_rows).to_csv(
            unit_dir / "answer_topic_stats_sentence_level.csv", index=False, encoding="utf-8-sig"
        )
    else:
        answer_out = answer_df.copy()
        answer_out["topic_id"] = labels_df["topic_id"].tolist()
        answer_out["has_valid_topic"] = answer_out["topic_id"] != OUTLIER_TOPIC_ID
        answer_out["answer_create_time"] = answer_out["answer_create_time"].astype(str)
        answer_out[["time_seq", "seq"]] = answer_out[["time_seq", "seq"]].astype("Int64")
        answer_out.to_csv(unit_dir / "answer_topic_labels_answer_level.csv", index=False, encoding="utf-8-sig")

    exported = {
        "run_ts_utc": utc_now_iso(),
        "unit": unit,
        "topic_audit_csv": str(unit_dir / f"{unit}_topic_audit.csv"),
        "text_samples_csv": str(unit_dir / f"{unit}_text_samples.csv"),
        "valid_topic_count": len(valid_topic_ids),
    }
    (unit_dir / "export_summary.json").write_text(
        json.dumps(exported, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[{unit}] exported {len(valid_topic_ids)} valid topics to {unit_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="BERTopic A/B topic modeling pipeline")
    parser.add_argument("--stage", choices=["fit", "export", "fit_export"], default="fit_export")
    parser.add_argument("--unit", choices=["sentence", "answer", "both"], default="both")
    args = parser.parse_args()

    if args.unit == "both":
        units = ["sentence", "answer"]
    else:
        units = [args.unit]

    if args.stage in {"fit", "fit_export"}:
        for unit in units:
            fit_unit(unit)
    if args.stage in {"export", "fit_export"}:
        for unit in units:
            export_unit(unit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
