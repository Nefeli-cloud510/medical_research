from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

ANSWER_PATH = RAW_DIR / "answer.parquet"
QUESTION_PATH = RAW_DIR / "question.parquet"
OUTPUT_CSV_PATH = TABLES_DIR / "04_title_tag_full_distribution_and_merge_suggestions.csv"
SUMMARY_PATH = AUDIT_DIR / "04_title_tag_review_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]


TITLE_RULES = {
    "主任医师": ("职称等级", "主任级", "merge_to_rank_level", "与主任中医师属于同一职称等级，可在等级变量中合并"),
    "主任中医师": ("职称等级", "主任级", "merge_to_rank_level", "与主任医师属于同一职称等级，可在等级变量中合并"),
    "副主任医师": ("职称等级", "副主任级", "merge_to_rank_level", "与副主任中医师属于同一职称等级，可在等级变量中合并"),
    "副主任中医师": ("职称等级", "副主任级", "merge_to_rank_level", "与副主任医师属于同一职称等级，可在等级变量中合并"),
    "主治医师": ("职称等级", "主治级", "merge_to_rank_level", "与主治中医师属于同一职称等级，可在等级变量中合并"),
    "主治中医师": ("职称等级", "主治级", "merge_to_rank_level", "与主治医师属于同一职称等级，可在等级变量中合并"),
    "住院医师": ("职称等级", "住院/执业医师级", "keep_or_merge_carefully", "可与医师合并为低年资医师组，但建议与医师做敏感性分析"),
    "医师": ("职称等级", "住院/执业医师级", "keep_or_merge_carefully", "可与住院医师合并为低年资医师组，但建议与住院医师做敏感性分析"),
    "助理医师": ("职称等级", "助理级", "keep_as_separate", "建议单列或并入最低等级医师组"),
    "主管护师": ("护理序列", "主管护师", "keep_as_nursing_series", "建议保留护理序列，不与医师职称直接合并"),
    "副主任护师": ("护理序列", "副主任护师", "keep_as_nursing_series", "建议保留护理序列，不与医师职称直接合并"),
    "主任护师": ("护理序列", "主任护师", "keep_as_nursing_series", "建议保留护理序列，不与医师职称直接合并"),
    "护师": ("护理序列", "护师", "keep_as_nursing_series", "建议保留护理序列，不与医师职称直接合并"),
    "护士": ("护理序列", "护士", "keep_as_nursing_series", "建议保留护理序列，不与医师职称直接合并"),
    "助产士": ("护理序列", "助产士", "keep_as_nursing_series", "建议保留护理/助产序列，不与医师职称直接合并"),
    "二级心理咨询师": ("心理咨询序列", "心理咨询师", "merge_within_series", "可与三级心理咨询师合并为心理咨询序列"),
    "三级心理咨询师": ("心理咨询序列", "心理咨询师", "merge_within_series", "可与二级心理咨询师合并为心理咨询序列"),
    "治疗师": ("医技序列", "治疗师", "keep_as_technician_series", "建议保留医技序列，不与医师职称直接合并"),
    "主管检验师": ("医技序列", "检验/技师", "merge_within_series", "建议并入医技序列"),
    "主管药师": ("药学序列", "药师", "merge_within_series", "可与药师/副主任药师合并为药学序列"),
    "副主任药师": ("药学序列", "药师", "merge_within_series", "可与药师/主管药师合并为药学序列"),
    "药师": ("药学序列", "药师", "merge_within_series", "可与副主任药师/主管药师合并为药学序列"),
    "主管技师": ("医技序列", "检验/技师", "merge_within_series", "建议并入医技序列"),
    "主任技师": ("医技序列", "检验/技师", "merge_within_series", "建议并入医技序列"),
    "副主任技师": ("医技序列", "检验/技师", "merge_within_series", "建议并入医技序列"),
    "二级营养师": ("营养序列", "营养师", "keep_as_separate_series", "建议单独保留营养序列"),
    "普通门诊": ("异常值", "普通门诊", "manual_review", "这更像门诊类型而非职称，建议人工核对来源"),
}

TAG_RULES = {
    "就医互助": ("平台通用标签", "", "drop_generic_tag", "平台通用标签，建议单独剥离，不作为疾病/症状主题"),
    "发烧": ("症状", "发热/发烧", "merge_synonym", "与发热语义相同，建议统一"),
    "发热": ("症状", "发热/发烧", "merge_synonym", "与发烧语义相同，建议统一"),
    "甲状腺结节": ("疾病/部位标签", "甲状腺结节", "keep_specific", "器官特异标签，建议保留，不与泛化结节直接合并"),
    "乳腺结节": ("疾病/部位标签", "乳腺结节", "keep_specific", "器官特异标签，建议保留，不与泛化结节直接合并"),
    "结节": ("泛化病理标签", "结节", "keep_with_hierarchy", "建议保留为上位标签，不直接并入甲状腺结节/乳腺结节"),
    "女性健康": ("主题标签", "女性健康", "manual_review", "与妇科高度相关，但更像主题分类，建议人工判断是否合并"),
    "妇科": ("科室/主题标签", "妇科", "manual_review", "与女性健康相关但不完全同义，建议人工判断是否合并"),
    "亲子育儿": ("主题标签", "亲子育儿", "manual_review", "与儿科相关但不完全同义，建议人工判断是否合并"),
    "儿科": ("科室/主题标签", "儿科", "manual_review", "与亲子育儿相关但不完全同义，建议人工判断是否合并"),
    "皮肤科": ("科室标签", "皮肤科", "manual_review", "与皮肤疾病相关但不完全同义，建议人工判断是否合并"),
    "皮肤疾病": ("疾病大类标签", "皮肤疾病", "manual_review", "与皮肤科相关但不完全同义，建议人工判断是否合并"),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
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
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def split_tags(tag_text: str) -> list[str]:
    parts = re.split(r"[|,，、;/；\s]+", tag_text)
    return [part.strip() for part in parts if part.strip()]


def title_suggestion(value: str) -> tuple[str, str, str, str]:
    if value in TITLE_RULES:
        return TITLE_RULES[value]
    return ("其他序列", value, "manual_review", "低频或未覆盖职称，建议人工复核后决定是否合并")


def tag_suggestion(value: str) -> tuple[str, str, str, str]:
    if value in TAG_RULES:
        return TAG_RULES[value]
    if value.endswith("癌"):
        return ("肿瘤标签", value, "keep_specific", "癌种标签建议保留，不直接合并到泛肿瘤标签")
    if value.endswith("结节"):
        return ("结节标签", value, "keep_specific", "部位特异结节标签建议保留，不直接并入泛化结节")
    if value.endswith("科"):
        return ("科室标签", value, "manual_review", "科室标签与疾病标签混杂，建议人工判断是否参与主题建模")
    return ("其他标签", value, "keep_as_is", "当前无高置信同义项，建议先保留")


def build_title_distribution() -> tuple[pd.DataFrame, dict]:
    title_series = pd.read_parquet(ANSWER_PATH, columns=["tech_title"])["tech_title"].map(clean_text)
    title_series = title_series[title_series != ""]
    counts = title_series.value_counts()
    total_non_missing = int(title_series.shape[0])

    rows = []
    for rank, (value, count) in enumerate(counts.items(), start=1):
        category, canonical, action, reason = title_suggestion(value)
        rows.append(
            {
                "field_name": "tech_title",
                "value_level": "raw_value",
                "rank": rank,
                "raw_value": value,
                "count": int(count),
                "share_in_field": round(float(count) / total_non_missing, 6),
                "field_non_missing_count": total_non_missing,
                "distinct_value_count": int(counts.shape[0]),
                "suggested_category": category,
                "suggested_canonical_value": canonical,
                "suggested_action": action,
                "suggestion_reason": reason,
            }
        )

    summary = {
        "non_missing_count": total_non_missing,
        "distinct_value_count": int(counts.shape[0]),
        "top_5": counts.head(5).to_dict(),
    }
    return pd.DataFrame(rows), summary


def build_tag_distribution() -> tuple[pd.DataFrame, dict]:
    tag_series = pd.read_parquet(QUESTION_PATH, columns=["tags"])["tags"].map(clean_text)
    tag_series = tag_series[tag_series != ""]
    token_counter: Counter[str] = Counter()

    for raw_value in tag_series:
        tokens = list(dict.fromkeys(split_tags(raw_value)))
        token_counter.update(tokens)

    total_non_missing_rows = int(tag_series.shape[0])
    total_token_occurrences = int(sum(token_counter.values()))
    counts = pd.Series(token_counter).sort_values(ascending=False)

    rows = []
    for rank, (value, count) in enumerate(counts.items(), start=1):
        category, canonical, action, reason = tag_suggestion(value)
        rows.append(
            {
                "field_name": "tags",
                "value_level": "split_tag",
                "rank": rank,
                "raw_value": value,
                "count": int(count),
                "share_in_field": round(float(count) / total_non_missing_rows, 6),
                "field_non_missing_count": total_non_missing_rows,
                "distinct_value_count": int(counts.shape[0]),
                "suggested_category": category,
                "suggested_canonical_value": canonical,
                "suggested_action": action,
                "suggestion_reason": reason,
            }
        )

    summary = {
        "non_missing_row_count": total_non_missing_rows,
        "token_occurrence_count": total_token_occurrences,
        "distinct_split_tag_count": int(counts.shape[0]),
        "top_10": counts.head(10).to_dict(),
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    ensure_directories()
    title_df, title_summary = build_title_distribution()
    tag_df, tag_summary = build_tag_distribution()
    output_df = pd.concat([title_df, tag_df], ignore_index=True)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only",
        "output_csv_path": str(OUTPUT_CSV_PATH),
        "tech_title_summary": title_summary,
        "tags_summary": tag_summary,
        "row_count": int(output_df.shape[0]),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "04_title_tag_review",
                "action": "export_title_and_tag_distribution_with_merge_suggestions",
                "status": "completed",
                "target_path": str(OUTPUT_CSV_PATH),
                "details": (
                    f"rows={summary['row_count']}; "
                    f"tech_title_distinct={title_summary['distinct_value_count']}; "
                    f"tag_distinct={tag_summary['distinct_split_tag_count']}"
                ),
            }
        ]
    )

    print(f"CSV written to {OUTPUT_CSV_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
