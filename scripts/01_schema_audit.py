from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
METADATA_DIR = ROOT_DIR / "metadata"
OUTPUTS_AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"
SCHEMA_MAPPING_PATH = METADATA_DIR / "schema_mapping.yml"
SUMMARY_PATH = OUTPUTS_AUDIT_DIR / "01_schema_audit_summary.json"
TYPE_MATRIX_PATH = OUTPUTS_AUDIT_DIR / "01_field_type_matrix.csv"
TARGET_RELATIVE_PATHS = (
    Path("data/raw/answer.parquet"),
    Path("data/raw/question.parquet"),
)

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

ID_FIELDS_OF_INTEREST = ("question_id",)
TEXT_FIELDS_OF_INTEREST = ("question_desc", "answer")
TIME_NAME_KEYWORDS = ("time", "date", "datetime", "publish", "release", "crawl")
QUESTION_PUNCT = ("?", "？")
FIELD_EXPLANATIONS = {
    "question_id": "例：100002518649569281，标定每条问题的唯一字段；question的主键，answer的外键",
    "user_id": "回答医生的平台账号id，唯一标定每位医生",
    "tech_title": "例：主治医师，每位医生的职称",
    "dept": "例：消化内科，每位医生所属的领域",
    "hospital": "每位医生的所属医院/单位",
    "answer": "具体作答的文本内容",
    "create_time": "回复/提问发布的具体时间",
    "thumb": "每条回答获得的点赞数",
    "help_status": "？",
    "seq": "？",
    "gender": "男/女，存在NULL值",
    "age": "存在NULL值",
    "address": "省/市 级，直辖市则精确到市",
    "question_desc": "字符串，具体问题的描述，呈现口语化特征，标点符号问号居多，符合问题的特征",
    "tags": "例：亲子育儿、咳嗽、感冒等，为具体症状/疾病的标签",
    "bounty": "？",
    "views": "？",
    "ans_count": "本条问题下回答的数量",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def append_cleaning_log(rows: list[dict[str, str]]) -> None:
    file_exists = CLEANING_LOG_PATH.exists()
    is_empty = (not file_exists) or CLEANING_LOG_PATH.stat().st_size == 0
    with CLEANING_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANING_LOG_COLUMNS)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)


def is_nested_type(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
        or pa.types.is_struct(data_type)
        or pa.types.is_map(data_type)
    )


def infer_time_representation(
    field_name: str,
    data_type: pa.DataType,
    sample_values: list[Any],
    is_time_candidate: bool,
) -> str:
    lowered = field_name.lower()
    if pa.types.is_timestamp(data_type):
        return "timestamp_type"
    if pa.types.is_date(data_type):
        return "date_type"
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return "string_time_candidate" if any(keyword in lowered for keyword in TIME_NAME_KEYWORDS) else "string"
    if pa.types.is_integer(data_type):
        if not is_time_candidate:
            return "int_non_timestamp_or_low_precision"
        if not sample_values:
            return "int_time_candidate_unknown_unit"
        ints = [abs(int(v)) for v in sample_values if v is not None]
        if not ints:
            return "int_time_candidate_unknown_unit"
        max_value = max(ints)
        digits = len(str(max_value))
        if digits >= 13:
            return "int_timestamp_ms_candidate"
        if digits >= 10:
            return "int_timestamp_s_candidate"
        return "int_non_timestamp_or_low_precision"
    return str(data_type)


def normalize_standard_name(field_name: str) -> str:
    name = field_name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def sample_column_values(file_path: Path, column_name: str, sample_size: int = 200) -> list[Any]:
    try:
        table = pq.read_table(file_path, columns=[column_name])
        if table.num_rows == 0:
            return []
        return table[column_name].slice(0, sample_size).to_pylist()
    except Exception:
        return []


def json_like_ratio(sample_values: list[Any]) -> float:
    if not sample_values:
        return 0.0
    checked = 0
    json_like = 0
    for value in sample_values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        checked += 1
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            json_like += 1
    if checked == 0:
        return 0.0
    return json_like / checked


def question_mark_ratio(sample_values: list[Any]) -> float:
    if not sample_values:
        return 0.0
    checked = 0
    marked = 0
    for value in sample_values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        checked += 1
        if any(mark in text for mark in QUESTION_PUNCT):
            marked += 1
    if checked == 0:
        return 0.0
    return marked / checked


def scan_files() -> list[Path]:
    files: list[Path] = []
    for relative_path in TARGET_RELATIVE_PATHS:
        candidate = ROOT_DIR / relative_path
        if candidate.exists():
            files.append(candidate)
    return sorted(files)


def build_mapping_yaml(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("version: 1")
    lines.append(f"updated_at_utc: '{summary['run_ts_utc']}'")
    lines.append("raw_root: data/raw")
    lines.append("policy:")
    lines.append("  raw_read_only: true")
    lines.append("  rename_in_code_forbidden: true")
    lines.append("  keep_source_column_name: true")
    lines.append("  notes: Use this mapping for renaming in downstream staging scripts only.")
    lines.append("id_fields_of_interest:")
    for field_name in ID_FIELDS_OF_INTEREST:
        lines.append(f"  - {field_name}")
    lines.append("files:")

    for file_item in summary["files"]:
        lines.append(f"  - file_name: {file_item['file_name']}")
        lines.append(f"    relative_path: {file_item['relative_path']}")
        lines.append("    column_mappings:")
        for column in file_item["columns"]:
            lines.append(f"      - source_name: {column['name']}")
            lines.append(f"        target_name: {normalize_standard_name(column['name'])}")
            lines.append(f"        source_type: '{column['type']}'")
            lines.append(f"        target_type: '{column['recommended_target_type']}'")
            lines.append(f"        is_id: {str(column['is_id']).lower()}")
            lines.append(f"        is_time_candidate: {str(column['is_time_candidate']).lower()}")
            lines.append(f"        time_representation: '{column['time_representation']}'")
            lines.append(f"        explaination: '{column['explaination']}'")
            lines.append(f"        json_like_ratio: {column['json_like_ratio']:.4f}")
            lines.append(f"        question_mark_ratio: {column['question_mark_ratio']:.4f}")
    return "\n".join(lines) + "\n"


def write_type_matrix(
    field_type_matrix: dict[str, dict[str, str]],
    field_order_index: dict[str, tuple[int, int]],
) -> None:
    rows: list[dict[str, str]] = []
    ordered_field_names = sorted(
        field_type_matrix,
        key=lambda field_name: field_order_index.get(field_name, (10**9, 10**9,)),
    )
    for field_name in ordered_field_names:
        type_by_file = field_type_matrix[field_name]
        distinct_types = sorted({v for v in type_by_file.values() if v})
        rows.append(
            {
                "field_name": field_name,
                "distinct_type_count": str(len(distinct_types)),
                "types": "|".join(distinct_types),
                "explaination": FIELD_EXPLANATIONS.get(field_name, ""),
                "files": "|".join(sorted(type_by_file.keys())),
            }
        )
    with TYPE_MATRIX_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["field_name", "distinct_type_count", "types", "explaination", "files"],
        )
        writer.writeheader()
        writer.writerows(rows)


def audit() -> dict[str, Any]:
    files = scan_files()
    file_summaries: list[dict[str, Any]] = []
    field_type_matrix: dict[str, dict[str, str]] = {}
    field_order_index: dict[str, tuple[int, int]] = {}
    id_presence = {key: [] for key in ID_FIELDS_OF_INTEREST}
    id_float_fields: list[dict[str, str]] = []
    nested_fields: list[dict[str, str]] = []
    text_json_candidates: list[dict[str, Any]] = []
    question_answer_mix_signals: list[dict[str, Any]] = []

    for file_idx, file_path in enumerate(files):
        relative_path = file_path.relative_to(ROOT_DIR).as_posix()
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema_arrow
        column_summaries: list[dict[str, Any]] = []
        file_has_create_time = "create_time" in schema.names
        file_has_release_time = "release_time" in schema.names

        for column_idx, field in enumerate(schema):
            field_name = field.name
            field_type = str(field.type)
            sample_values = sample_column_values(file_path, field_name)
            field_order_index.setdefault(field_name, (file_idx, column_idx))
            is_id = field_name.lower().endswith("_id")
            is_time_candidate = (
                any(keyword in field_name.lower() for keyword in TIME_NAME_KEYWORDS)
                or pa.types.is_timestamp(field.type)
                or pa.types.is_date(field.type)
            )
            time_representation = infer_time_representation(
                field_name,
                field.type,
                sample_values,
                is_time_candidate,
            )
            json_ratio = json_like_ratio(sample_values) if field_name in TEXT_FIELDS_OF_INTEREST else 0.0
            qmark_ratio = question_mark_ratio(sample_values) if field_name in TEXT_FIELDS_OF_INTEREST else 0.0

            column_summaries.append(
                {
                    "name": field_name,
                    "type": field_type,
                    "is_id": is_id,
                    "is_time_candidate": is_time_candidate,
                    "time_representation": time_representation,
                    "explaination": FIELD_EXPLANATIONS.get(field_name, ""),
                    "json_like_ratio": json_ratio,
                    "question_mark_ratio": qmark_ratio,
                    "recommended_target_type": field_type,
                }
            )

            field_type_matrix.setdefault(field_name, {})[relative_path] = field_type

            if field_name in ID_FIELDS_OF_INTEREST:
                id_presence[field_name].append(relative_path)
            if is_id and pa.types.is_floating(field.type):
                id_float_fields.append(
                    {"file": relative_path, "field": field_name, "type": field_type}
                )

            if is_nested_type(field.type):
                nested_fields.append({"file": relative_path, "field": field_name, "type": field_type})

            if field_name in TEXT_FIELDS_OF_INTEREST and json_ratio >= 0.2:
                text_json_candidates.append(
                    {
                        "file": relative_path,
                        "field": field_name,
                        "json_like_ratio": round(json_ratio, 4),
                    }
                )

            if field_name in ("question_desc", "answer"):
                question_answer_mix_signals.append(
                    {
                        "file": relative_path,
                        "field": field_name,
                        "question_mark_ratio": round(qmark_ratio, 4),
                    }
                )

        file_summaries.append(
            {
                "file_name": file_path.name,
                "relative_path": relative_path,
                "row_count": parquet_file.metadata.num_rows,
                "row_group_count": parquet_file.metadata.num_row_groups,
                "has_create_time": file_has_create_time,
                "has_release_time": file_has_release_time,
                "columns": column_summaries,
            }
        )

    field_type_conflicts = []
    for field_name, by_file in field_type_matrix.items():
        types = sorted(set(by_file.values()))
        if len(types) > 1:
            field_type_conflicts.append(
                {
                    "field": field_name,
                    "types": types,
                    "files": sorted(by_file.keys()),
                }
            )

    time_confusion_candidates = [
        {
            "relative_path": item["relative_path"],
            "has_create_time": item["has_create_time"],
            "has_release_time": item["has_release_time"],
            "note": (
                "both_present_compare_semantics"
                if item["has_create_time"] and item["has_release_time"]
                else "single_time_field_check_business_definition"
            ),
        }
        for item in file_summaries
        if item["has_create_time"] or item["has_release_time"]
    ]

    summary = {
        "run_ts_utc": utc_now_iso(),
        "raw_dir": str(RAW_DIR),
        "scope": "answer_question_only",
        "file_count": len(files),
        "id_presence": {k: sorted(v) for k, v in id_presence.items()},
        "missing_id_fields": [k for k, v in id_presence.items() if not v],
        "id_float_fields": id_float_fields,
        "nested_fields": nested_fields,
        "text_json_candidates": text_json_candidates,
        "field_type_conflicts": field_type_conflicts,
        "question_answer_mix_signals": question_answer_mix_signals,
        "time_confusion_candidates": time_confusion_candidates,
        "files": file_summaries,
    }

    write_type_matrix(field_type_matrix, field_order_index)
    return summary


def cleaning_log_entries(summary: dict[str, Any]) -> list[dict[str, str]]:
    ts = summary["run_ts_utc"]
    return [
        {
            "run_ts_utc": ts,
            "step": "01_schema_audit",
            "action": "scan_schema_and_types",
            "status": "completed",
            "target_path": str(RAW_DIR),
            "details": (
                f"files={summary['file_count']}; "
                f"missing_ids={','.join(summary['missing_id_fields']) or 'none'}; "
                f"type_conflicts={len(summary['field_type_conflicts'])}; "
                f"scope={summary['scope']}"
            ),
        },
        {
            "run_ts_utc": ts,
            "step": "01_schema_audit",
            "action": "write_schema_mapping",
            "status": "completed",
            "target_path": str(SCHEMA_MAPPING_PATH),
            "details": "preserved_source_columns=true",
        },
        {
            "run_ts_utc": ts,
            "step": "01_schema_audit",
            "action": "write_schema_audit_outputs",
            "status": "completed",
            "target_path": str(SUMMARY_PATH),
            "details": f"type_matrix={TYPE_MATRIX_PATH.name}",
        },
    ]


def main() -> None:
    ensure_directories()
    summary = audit()
    SCHEMA_MAPPING_PATH.write_text(build_mapping_yaml(summary), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    append_cleaning_log(cleaning_log_entries(summary))

    print(f"Schema audit files scanned: {summary['file_count']}")
    print(f"Schema mapping written: {SCHEMA_MAPPING_PATH}")
    print(f"Schema audit summary written: {SUMMARY_PATH}")
    print(f"Field type matrix written: {TYPE_MATRIX_PATH}")
    print(f"Missing id fields: {summary['missing_id_fields']}")
    print(f"ID float fields: {len(summary['id_float_fields'])}")
    print(f"Nested fields: {len(summary['nested_fields'])}")


if __name__ == "__main__":
    main()
