from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"
AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

INPUT_PATH = ANALYSIS_DIR / "major5_text_cleaning_prep_seq_clean.parquet"
OUTPUT_ANSWERS_PATH = ANALYSIS_DIR / "major5_stratified_question_samples_answers.parquet"
QUESTION_MANIFEST_PATH = TABLES_DIR / "18_major5_stratified_question_samples_question_manifest.csv"
FLOW_QUESTION_IDS_PATH = TABLES_DIR / "18_flow_run_question_ids.csv"
MEASUREMENT_QUESTION_IDS_PATH = TABLES_DIR / "18_measurement_dev_question_ids.csv"
VALIDATION_QUESTION_IDS_PATH = TABLES_DIR / "18_independent_manual_validation_question_ids.csv"
SUMMARY_PATH = AUDIT_DIR / "18_major5_stratified_question_samples_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

GLOBAL_RANDOM_SEED = 20261006

QUESTION_READ_COLUMNS = [
    "question_id",
    "question_create_time",
    "dept",
    "ans_count",
]

ANSWER_READ_COLUMNS = [
    "question_id",
    "seq",
    "user_id",
    "tech_title",
    "dept",
    "hospital",
    "answer",
    "thumb",
    "help_status",
    "answer_create_time",
    "question_desc",
    "tags",
    "bounty",
    "views",
    "ans_count",
    "question_create_time",
    "time_seq",
    "observed_seq_by_time",
    "prior_answer_count",
    "prior_answer_count_by_time",
    "answer_delay_hours",
    "is_answer_time_valid",
    "is_seq_time_aligned",
    "time_rank_within_question",
    "has_prior_answers",
    "hospitallevel",
    "hospital_key",
    "hospitallevel_final",
    "hospitallevel_fill_method",
    "hospitallevel_source",
    "hospitallevel_note",
    "manual_hospitallevel",
    "manual_hospitallevel_source",
    "manual_hospitallevel_note",
    "has_round2_override",
    "question_answer_join_ready",
    "question_user_answer_count",
    "doctor_answer_order_within_question",
]

ANSWER_COUNT_BINS = [0, 1, 3, 5, 10, 20, 10**9]
ANSWER_COUNT_BAND_LABELS = ["1", "2-3", "4-5", "6-10", "11-20", "21+"]


@dataclass(frozen=True)
class SampleConfig:
    sample_layer: str
    target_question_count: int
    purpose: str


SAMPLE_CONFIGS = [
    SampleConfig(
        sample_layer="flow_run",
        target_question_count=1_000,
        purpose="流程试跑：检查代码、顺序、缺失与输出结构。",
    ),
    SampleConfig(
        sample_layer="measurement_dev",
        target_question_count=5_000,
        purpose="测量开发：调试清洗规则、主题模型与指标选择。",
    ),
    SampleConfig(
        sample_layer="independent_manual_validation",
        target_question_count=1_000,
        purpose="独立人工验证：与开发集不重叠，用于人工判断指标是否反映新增知识。",
    ),
]


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


def deterministic_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest[:8], 16)


def normalize_dept(value: object) -> str:
    if pd.isna(value):
        return "[missing_dept]"
    text = str(value).strip()
    return text or "[missing_dept]"


def build_question_frame() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH, columns=QUESTION_READ_COLUMNS)
    question_df = (
        df.groupby("question_id", as_index=False)
        .agg(
            question_create_time=("question_create_time", "first"),
            dept=("dept", "first"),
            ans_count=("ans_count", "first"),
        )
        .copy()
    )

    question_df["question_month"] = (
        pd.to_datetime(question_df["question_create_time"]).dt.to_period("M").astype(str)
    )
    question_df["dept_strata"] = question_df["dept"].map(normalize_dept)
    question_df["ans_count"] = question_df["ans_count"].astype("int64")
    question_df["answer_count_band"] = pd.cut(
        question_df["ans_count"],
        bins=ANSWER_COUNT_BINS,
        labels=ANSWER_COUNT_BAND_LABELS,
        right=True,
        include_lowest=True,
    ).astype("string")
    return question_df


def allocate_counts(counts: pd.Series, target_total: int) -> pd.Series:
    counts = counts.astype("int64")
    available_total = int(counts.sum())
    target_total = min(int(target_total), available_total)
    if target_total <= 0 or available_total <= 0:
        return pd.Series(0, index=counts.index, dtype="int64")

    raw = counts / available_total * target_total
    allocated = raw.apply(int).astype("int64")
    allocated = allocated.clip(upper=counts)
    remaining = target_total - int(allocated.sum())

    while remaining > 0:
        eligible_mask = allocated < counts
        if not bool(eligible_mask.any()):
            break
        remainder = (raw - allocated).where(eligible_mask, other=-1.0)
        eligible_order = (
            pd.DataFrame(
                {
                    "remainder": remainder,
                    "available": counts - allocated,
                }
            )
            .sort_values(["remainder", "available"], ascending=[False, False], kind="stable")
            .index.tolist()
        )
        incremented = 0
        for key in eligible_order:
            if remaining <= 0:
                break
            if allocated.loc[key] >= counts.loc[key]:
                continue
            allocated.loc[key] += 1
            remaining -= 1
            incremented += 1
        if incremented == 0:
            break

    return allocated.astype("int64")


def sample_rows_from_group(group_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(int(n), len(group_df))
    if n <= 0:
        return group_df.iloc[0:0].copy()
    return group_df.sample(n=n, random_state=seed, replace=False)


def hierarchical_stratified_sample(
    pool_df: pd.DataFrame,
    target_question_count: int,
    sample_layer: str,
) -> pd.DataFrame:
    target_question_count = min(int(target_question_count), len(pool_df))
    if target_question_count <= 0:
        return pool_df.iloc[0:0].copy()

    selected_parts: list[pd.DataFrame] = []
    month_counts = pool_df["question_month"].value_counts(sort=False)
    month_alloc = allocate_counts(month_counts, target_question_count)

    for month, month_n in month_alloc.items():
        if month_n <= 0:
            continue
        month_df = pool_df.loc[pool_df["question_month"] == month].copy()
        dept_counts = month_df["dept_strata"].value_counts(sort=False)
        dept_alloc = allocate_counts(dept_counts, int(month_n))

        for dept_strata, dept_n in dept_alloc.items():
            if dept_n <= 0:
                continue
            dept_df = month_df.loc[month_df["dept_strata"] == dept_strata].copy()
            band_counts = dept_df["answer_count_band"].value_counts(sort=False)
            band_alloc = allocate_counts(band_counts, int(dept_n))

            for answer_count_band, band_n in band_alloc.items():
                if band_n <= 0:
                    continue
                band_df = dept_df.loc[dept_df["answer_count_band"] == answer_count_band].copy()
                seed = deterministic_seed(
                    GLOBAL_RANDOM_SEED,
                    sample_layer,
                    month,
                    dept_strata,
                    answer_count_band,
                    band_n,
                )
                selected_parts.append(sample_rows_from_group(band_df, int(band_n), seed))

    sampled_df = pd.concat(selected_parts, ignore_index=True) if selected_parts else pool_df.iloc[0:0].copy()
    sampled_df = sampled_df.drop_duplicates(subset=["question_id"], keep="first").copy()

    if len(sampled_df) < target_question_count:
        needed = target_question_count - len(sampled_df)
        remaining_pool = pool_df.loc[~pool_df["question_id"].isin(sampled_df["question_id"])].copy()
        fallback_seed = deterministic_seed(GLOBAL_RANDOM_SEED, sample_layer, "fallback_fill", needed)
        sampled_df = pd.concat(
            [sampled_df, sample_rows_from_group(remaining_pool, needed, fallback_seed)],
            ignore_index=True,
        )

    sampled_df = sampled_df.sample(
        frac=1.0,
        random_state=deterministic_seed(GLOBAL_RANDOM_SEED, sample_layer, "final_shuffle"),
    ).reset_index(drop=True)
    sampled_df["sample_draw_order"] = sampled_df.index + 1
    sampled_df["sample_layer"] = sample_layer
    sampled_df["sample_seed"] = GLOBAL_RANDOM_SEED
    return sampled_df


def write_question_ids(path: Path, question_ids: pd.Series) -> None:
    question_ids.to_frame(name="question_id").to_csv(path, index=False, encoding="utf-8-sig")


def build_layer_summary(layer_df: pd.DataFrame, answer_rows: int) -> dict[str, object]:
    month_distribution = (
        layer_df["question_month"]
        .value_counts()
        .sort_index()
        .head(24)
        .rename_axis("question_month")
        .reset_index(name="question_count")
        .to_dict(orient="records")
    )
    answer_band_distribution = (
        layer_df["answer_count_band"]
        .value_counts()
        .reindex(ANSWER_COUNT_BAND_LABELS, fill_value=0)
        .rename_axis("answer_count_band")
        .reset_index(name="question_count")
        .to_dict(orient="records")
    )
    top_depts = (
        layer_df["dept_strata"]
        .value_counts()
        .head(20)
        .rename_axis("dept")
        .reset_index(name="question_count")
        .to_dict(orient="records")
    )

    return {
        "sampled_question_count": int(len(layer_df)),
        "sampled_answer_rows": int(answer_rows),
        "month_distribution_preview": month_distribution,
        "answer_count_band_distribution": answer_band_distribution,
        "top_departments_preview": top_depts,
    }


def main() -> None:
    ensure_directories()

    question_df = build_question_frame()
    remaining_pool = question_df.copy()
    sampled_question_parts: list[pd.DataFrame] = []

    for config in SAMPLE_CONFIGS:
        layer_sample_df = hierarchical_stratified_sample(
            remaining_pool,
            target_question_count=config.target_question_count,
            sample_layer=config.sample_layer,
        )
        sampled_question_parts.append(layer_sample_df)
        remaining_pool = remaining_pool.loc[
            ~remaining_pool["question_id"].isin(layer_sample_df["question_id"])
        ].reset_index(drop=True)

    sample_question_df = pd.concat(sampled_question_parts, ignore_index=True)
    sample_question_df["question_create_time"] = pd.to_datetime(
        sample_question_df["question_create_time"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    sample_question_df = sample_question_df[
        [
            "sample_layer",
            "sample_draw_order",
            "sample_seed",
            "question_id",
            "question_create_time",
            "question_month",
            "dept",
            "dept_strata",
            "ans_count",
            "answer_count_band",
        ]
    ].sort_values(["sample_layer", "sample_draw_order"], kind="stable")
    sample_question_df.to_csv(QUESTION_MANIFEST_PATH, index=False, encoding="utf-8-sig")

    flow_ids = sample_question_df.loc[sample_question_df["sample_layer"] == "flow_run", "question_id"]
    measurement_ids = sample_question_df.loc[
        sample_question_df["sample_layer"] == "measurement_dev", "question_id"
    ]
    validation_ids = sample_question_df.loc[
        sample_question_df["sample_layer"] == "independent_manual_validation", "question_id"
    ]
    write_question_ids(FLOW_QUESTION_IDS_PATH, flow_ids)
    write_question_ids(MEASUREMENT_QUESTION_IDS_PATH, measurement_ids)
    write_question_ids(VALIDATION_QUESTION_IDS_PATH, validation_ids)

    answer_df = pd.read_parquet(INPUT_PATH, columns=ANSWER_READ_COLUMNS).copy()
    answer_sample_df = answer_df.merge(
        sample_question_df[["question_id", "sample_layer", "sample_draw_order"]],
        on="question_id",
        how="inner",
    )
    answer_sample_df = answer_sample_df.sort_values(
        ["sample_layer", "sample_draw_order", "question_id", "time_seq", "answer_create_time", "seq"],
        kind="stable",
    ).reset_index(drop=True)
    answer_sample_df.to_parquet(OUTPUT_ANSWERS_PATH, index=False)

    overlap_checks = {
        "flow_vs_measurement_overlap_questions": int(len(set(flow_ids) & set(measurement_ids))),
        "flow_vs_validation_overlap_questions": int(len(set(flow_ids) & set(validation_ids))),
        "measurement_vs_validation_overlap_questions": int(len(set(measurement_ids) & set(validation_ids))),
    }

    layer_summaries: dict[str, object] = {}
    for config in SAMPLE_CONFIGS:
        layer_question_df = sample_question_df.loc[sample_question_df["sample_layer"] == config.sample_layer].copy()
        layer_answer_rows = int(
            answer_sample_df.loc[answer_sample_df["sample_layer"] == config.sample_layer].shape[0]
        )
        layer_summaries[config.sample_layer] = {
            "target_question_count": config.target_question_count,
            "purpose": config.purpose,
            **build_layer_summary(layer_question_df, layer_answer_rows),
        }

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "major5_stratified_question_samples",
        "input_path": str(INPUT_PATH),
        "output_answers_path": str(OUTPUT_ANSWERS_PATH),
        "question_manifest_path": str(QUESTION_MANIFEST_PATH),
        "sampling_principles": {
            "sample_unit": "question_id",
            "preserve_all_answers_within_sampled_question": True,
            "sampling_reason": "不能直接随机抽回答行，否则会破坏题内前序历史。",
            "stratification_dimensions": ["question_month", "dept_strata", "answer_count_band"],
            "answer_count_band_definition": {
                "1": "ans_count = 1",
                "2-3": "2 <= ans_count <= 3",
                "4-5": "4 <= ans_count <= 5",
                "6-10": "6 <= ans_count <= 10",
                "11-20": "11 <= ans_count <= 20",
                "21+": "ans_count >= 21",
            },
            "global_random_seed": GLOBAL_RANDOM_SEED,
            "sample_layers_are_disjoint": True,
        },
        "population": {
            "total_questions": int(len(question_df)),
            "total_answer_rows": int(pd.read_parquet(INPUT_PATH, columns=["question_id"]).shape[0]),
            "question_month_count": int(question_df["question_month"].nunique()),
            "department_count": int(question_df["dept_strata"].nunique()),
        },
        "layer_summaries": layer_summaries,
        "overlap_checks": overlap_checks,
        "question_id_outputs": {
            "flow_run": str(FLOW_QUESTION_IDS_PATH),
            "measurement_dev": str(MEASUREMENT_QUESTION_IDS_PATH),
            "independent_manual_validation": str(VALIDATION_QUESTION_IDS_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05e_build_major5_stratified_question_samples",
                "action": "sample_questions_by_month_dept_answercount_and_keep_full_threads",
                "status": "completed",
                "target_path": str(OUTPUT_ANSWERS_PATH),
                "details": (
                    f"seed={GLOBAL_RANDOM_SEED}; flow={len(flow_ids)}; "
                    f"measurement={len(measurement_ids)}; validation={len(validation_ids)}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05e_build_major5_stratified_question_samples",
                "action": "write_sample_question_manifests",
                "status": "completed",
                "target_path": str(QUESTION_MANIFEST_PATH),
                "details": (
                    f"flow_ids={FLOW_QUESTION_IDS_PATH.name}; "
                    f"measurement_ids={MEASUREMENT_QUESTION_IDS_PATH.name}; "
                    f"validation_ids={VALIDATION_QUESTION_IDS_PATH.name}"
                ),
            },
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "05e_build_major5_stratified_question_samples",
                "action": "write_sample_summary",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": (
                    f"overlap_flow_measurement={overlap_checks['flow_vs_measurement_overlap_questions']}; "
                    f"overlap_flow_validation={overlap_checks['flow_vs_validation_overlap_questions']}; "
                    f"overlap_measurement_validation={overlap_checks['measurement_vs_validation_overlap_questions']}"
                ),
            },
        ]
    )

    print(f"Sample question manifest written to {QUESTION_MANIFEST_PATH}")
    print(f"Sample answers written to {OUTPUT_ANSWERS_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
