from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
METADATA_DIR = ROOT_DIR / "metadata"
OUTPUTS_DIR = ROOT_DIR / "outputs"
AUDIT_DIR = OUTPUTS_DIR / "audit"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"
SUMMARY_PATH = AUDIT_DIR / "02_eda_summary.json"
SAMPLE_METRICS_PATH = TABLES_DIR / "02_sample_metrics.csv"
PATIENT_FEATURES_PATH = TABLES_DIR / "02_patient_features.csv"
DESCRIPTIVE_RELATIONSHIPS_PATH = TABLES_DIR / "02_descriptive_relationships.csv"
WORD_FREQUENCY_PATH = TABLES_DIR / "02_word_frequency.csv"
BATCH_SIZE = 100_000
WORD_SAMPLE_LIMIT = 50_000

QUESTION_PATH = RAW_DIR / "question.parquet"
ANSWER_PATH = RAW_DIR / "answer.parquet"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

PALETTE = {
    "blue": "#2F6BFF",
    "blue_light": "#6AA7FF",
    "blue_soft": "#A7CCFF",
    "cyan": "#57C7FF",
    "pink": "#F272B6",
    "orange": "#FFB255",
    "purple": "#8F6AF6",
    "gray": "#8E9AAF",
    "grid": "#E8EDF5",
    "bg": "#F7F9FC",
    "text": "#283247",
}

STOPWORDS = {
    "",
    "的",
    "了",
    "和",
    "是",
    "我",
    "也",
    "就",
    "都",
    "在",
    "有",
    "吗",
    "呢",
    "啊",
    "哦",
    "请问",
    "你好",
    "您好",
    "医生",
    "建议",
    "这种",
    "这个",
    "那个",
    "需要",
    "可以",
    "因为",
    "就是",
    "还是",
    "如果",
    "已经",
    "目前",
    "没有",
    "我们",
    "你",
    "他",
    "她",
    "它",
    "一个",
    "一下",
    "情况",
    "谢谢",
    "是否",
    "进行",
    "出现",
    "什么",
    "怎么",
    "多久",
    "多少",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_directories() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def append_cleaning_log(rows: list[dict[str, str]]) -> None:
    file_exists = CLEANING_LOG_PATH.exists()
    is_empty = (not file_exists) or CLEANING_LOG_PATH.stat().st_size == 0
    with CLEANING_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEANING_LOG_COLUMNS)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = PALETTE["bg"]
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = PALETTE["grid"]
    plt.rcParams["axes.labelcolor"] = PALETTE["text"]
    plt.rcParams["axes.titlecolor"] = PALETTE["text"]
    plt.rcParams["xtick.color"] = PALETTE["text"]
    plt.rcParams["ytick.color"] = PALETTE["text"]
    plt.rcParams["grid.color"] = PALETTE["grid"]
    plt.rcParams["savefig.facecolor"] = PALETTE["bg"]
    plt.rcParams["savefig.bbox"] = "tight"


def apply_axis_style(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["grid"])
    ax.spines["bottom"].set_color(PALETTE["grid"])
    ax.tick_params(axis="x", labelrotation=35)


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def text_length(value: object) -> int:
    text = clean_text(value)
    return len(text)


def to_month_string(series: pd.Series) -> pd.Series:
    return series.dt.to_period("M").astype(str)


def split_tags(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    parts = re.split(r"[|,，、;/；\s]+", text)
    return [part.strip() for part in parts if part.strip()]


def get_tokenizer():
    try:
        import jieba  # type: ignore

        return lambda text: jieba.lcut(text, cut_all=False)
    except Exception:
        return None


def fallback_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"https?://\S+", " ", text)
    cleaned = re.sub(r"[0-9]+", " ", cleaned)
    chunks = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z]+", cleaned)
    tokens: list[str] = []
    for chunk in chunks:
        if re.fullmatch(r"[A-Za-z]+", chunk):
            tokens.append(chunk.lower())
            continue
        if len(chunk) <= 4:
            tokens.append(chunk)
            continue
        tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return tokens


def tokenize_texts(texts: Iterable[str], sample_limit: int) -> Counter[str]:
    tokenizer = get_tokenizer()
    counter: Counter[str] = Counter()
    used = 0
    for text in texts:
        cleaned = clean_text(text)
        if not cleaned:
            continue
        used += 1
        if used > sample_limit:
            break
        raw_tokens = tokenizer(cleaned) if tokenizer else fallback_tokens(cleaned)
        for token in raw_tokens:
            token = clean_text(token).lower()
            if len(token) <= 1:
                continue
            if token in STOPWORDS:
                continue
            if re.fullmatch(r"[\W_]+", token):
                continue
            counter[token] += 1
    return counter


def summarize_counter(counter: Counter[str], name: str, top_n: int = 20) -> pd.DataFrame:
    rows = [{"label": key, name: value} for key, value in counter.most_common(top_n)]
    return pd.DataFrame(rows)


def percentile_clip(values: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    non_null = values[~np.isnan(values)]
    if non_null.size == 0:
        return non_null
    upper = np.quantile(non_null, quantile)
    return non_null[non_null <= upper]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_sample_and_thread(
    sample_metrics: dict[str, float],
    seq_counts: pd.Series,
    answer_count_by_question: pd.Series,
    thumb_values: np.ndarray,
) -> list[str]:
    figure_path = FIGURES_DIR / "02_sample_and_thread.png"
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("样本规模与线程结构", fontsize=18, fontweight="bold", color=PALETTE["text"])

    metrics_labels = ["问题数", "回答数", "医生数", "平均每问回答数"]
    metrics_values = [
        sample_metrics["question_count"],
        sample_metrics["answer_count"],
        sample_metrics["doctor_count"],
        sample_metrics["avg_answers_per_question"],
    ]
    axes[0, 0].bar(metrics_labels, metrics_values, color=[PALETTE["blue"], PALETTE["cyan"], PALETTE["purple"], PALETTE["orange"]])
    axes[0, 0].set_title("样本规模")
    apply_axis_style(axes[0, 0])

    seq_plot = seq_counts.sort_index().head(15)
    axes[0, 1].bar(seq_plot.index.astype(str), seq_plot.values, color=PALETTE["blue"])
    axes[0, 1].set_title("回答位次分布（前15位次）")
    axes[0, 1].set_xlabel("seq")
    apply_axis_style(axes[0, 1])

    thread_counts = answer_count_by_question.value_counts().sort_index()
    thread_plot = thread_counts.head(20)
    axes[1, 0].bar(thread_plot.index.astype(str), thread_plot.values, color=PALETTE["blue_light"])
    axes[1, 0].set_title("每个问题下回答数分布")
    axes[1, 0].set_xlabel("回答数")
    apply_axis_style(axes[1, 0])

    seq2_share = sample_metrics["answer_share_seq_ge_2"]
    question_multi_share = sample_metrics["question_share_with_multi_answers"]
    axes[1, 1].pie(
        [question_multi_share, 1 - question_multi_share],
        labels=["问题有2条及以上回答", "仅1条回答"],
        autopct="%1.1f%%",
        startangle=90,
        colors=[PALETTE["blue"], PALETTE["blue_soft"]],
        wedgeprops={"width": 0.45},
    )
    axes[1, 1].set_title(f"多回答问题占比\n回答样本中 seq>=2 占比 {seq2_share:.1%}")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    thumb_path = FIGURES_DIR / "02_thumb_distribution.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    thumb_plot = percentile_clip(thumb_values.astype(float), 0.99)
    ax.hist(thumb_plot, bins=40, color=PALETTE["blue"], edgecolor="white")
    ax.set_title("回答点赞分布（截到99分位）")
    ax.set_xlabel("thumb")
    ax.set_ylabel("频数")
    apply_axis_style(ax)
    fig.tight_layout()
    fig.savefig(thumb_path, dpi=220)
    plt.close(fig)
    return [figure_path.name, thumb_path.name]


def plot_structure_distributions(
    dept_top: pd.Series,
    tech_title_top: pd.Series,
    hospital_top: pd.Series,
) -> list[str]:
    figure_path = FIGURES_DIR / "02_structure_distributions.png"
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle("结构分布：科室、职称与医院集中度", fontsize=18, fontweight="bold", color=PALETTE["text"])

    axes[0].barh(dept_top.index[::-1], dept_top.values[::-1], color=PALETTE["blue"])
    axes[0].set_title("Top 15 科室")
    apply_axis_style(axes[0])

    axes[1].barh(tech_title_top.index[::-1], tech_title_top.values[::-1], color=PALETTE["purple"])
    axes[1].set_title("职称分布")
    apply_axis_style(axes[1])

    axes[2].barh(hospital_top.index[::-1], hospital_top.values[::-1], color=PALETTE["cyan"])
    axes[2].set_title("Top 15 医院")
    apply_axis_style(axes[2])

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return [figure_path.name]


def plot_text_and_time(
    question_lengths: np.ndarray,
    answer_lengths: np.ndarray,
    monthly_question_counts: pd.Series,
    monthly_answer_counts: pd.Series,
    response_delay_hours: np.ndarray,
) -> list[str]:
    figure_path = FIGURES_DIR / "02_text_and_time.png"
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("文本长度与时间分布", fontsize=18, fontweight="bold", color=PALETTE["text"])

    q_plot = percentile_clip(question_lengths.astype(float), 0.99)
    a_plot = percentile_clip(answer_lengths.astype(float), 0.99)
    axes[0, 0].hist(q_plot, bins=40, alpha=0.75, color=PALETTE["blue"], label="问题文本")
    axes[0, 0].hist(a_plot, bins=40, alpha=0.55, color=PALETTE["pink"], label="回答文本")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("文本长度分布（截到99分位）")
    axes[0, 0].set_xlabel("字符数")
    apply_axis_style(axes[0, 0])

    axes[0, 1].plot(monthly_question_counts.index, monthly_question_counts.values, color=PALETTE["blue"], linewidth=2.2, label="问题数")
    axes[0, 1].plot(monthly_answer_counts.index, monthly_answer_counts.values, color=PALETTE["pink"], linewidth=2.2, label="回答数")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].set_title("月度提问与回答数量趋势")
    axes[0, 1].set_xlabel("月份")
    apply_axis_style(axes[0, 1])

    delay_days = percentile_clip((response_delay_hours / 24).astype(float), 0.99)
    axes[1, 0].hist(delay_days, bins=40, color=PALETTE["cyan"], edgecolor="white")
    axes[1, 0].set_title("回答延迟分布（天，截到99分位）")
    axes[1, 0].set_xlabel("天数")
    apply_axis_style(axes[1, 0])

    summary_labels = ["问题中位长度", "回答中位长度", "延迟中位数(小时)", "延迟95分位(小时)"]
    summary_values = [
        np.nanmedian(question_lengths),
        np.nanmedian(answer_lengths),
        np.nanmedian(response_delay_hours),
        np.nanquantile(response_delay_hours[~np.isnan(response_delay_hours)], 0.95),
    ]
    axes[1, 1].bar(summary_labels, summary_values, color=[PALETTE["blue"], PALETTE["pink"], PALETTE["cyan"], PALETTE["orange"]])
    axes[1, 1].set_title("文本与响应速度摘要")
    apply_axis_style(axes[1, 1])

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return [figure_path.name]


def plot_patient_and_tags(
    gender_counts: pd.Series,
    age_values: np.ndarray,
    address_top: pd.Series,
    tag_top: pd.Series,
) -> list[str]:
    figure_path = FIGURES_DIR / "02_patient_and_tags.png"
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("患者特征与标签分布", fontsize=18, fontweight="bold", color=PALETTE["text"])

    axes[0, 0].bar(gender_counts.index.astype(str), gender_counts.values, color=[PALETTE["blue"], PALETTE["pink"], PALETTE["gray"]][: len(gender_counts)])
    axes[0, 0].set_title("性别分布")
    apply_axis_style(axes[0, 0])

    age_plot = percentile_clip(age_values.astype(float), 0.99)
    axes[0, 1].hist(age_plot, bins=30, color=PALETTE["orange"], edgecolor="white")
    axes[0, 1].set_title("年龄分布（截到99分位）")
    axes[0, 1].set_xlabel("年龄")
    apply_axis_style(axes[0, 1])

    axes[1, 0].barh(address_top.index[::-1], address_top.values[::-1], color=PALETTE["cyan"])
    axes[1, 0].set_title("Top 15 地区")
    apply_axis_style(axes[1, 0])

    axes[1, 1].barh(tag_top.index[::-1], tag_top.values[::-1], color=PALETTE["purple"])
    axes[1, 1].set_title("Top 15 标签")
    apply_axis_style(axes[1, 1])

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return [figure_path.name]


def plot_word_frequencies(question_top: pd.Series, answer_top: pd.Series) -> list[str]:
    figure_path = FIGURES_DIR / "02_word_frequencies.png"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("问题文本与回答文本高频词", fontsize=18, fontweight="bold", color=PALETTE["text"])

    axes[0].barh(question_top.index[::-1], question_top.values[::-1], color=PALETTE["blue"])
    axes[0].set_title("问题文本 Top 20 高频词")
    apply_axis_style(axes[0])

    axes[1].barh(answer_top.index[::-1], answer_top.values[::-1], color=PALETTE["pink"])
    axes[1].set_title("回答文本 Top 20 高频词")
    apply_axis_style(axes[1])

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return [figure_path.name]


def plot_descriptive_relationships(
    thumb_by_title: pd.Series,
    thumb_by_seq: pd.Series,
    answer_len_by_dept: pd.Series,
    bounty_group_ans_count: pd.Series,
) -> list[str]:
    figure_path = FIGURES_DIR / "02_descriptive_relationships.png"
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("描述性关系", fontsize=18, fontweight="bold", color=PALETTE["text"])

    axes[0, 0].barh(thumb_by_title.index[::-1], thumb_by_title.values[::-1], color=PALETTE["blue"])
    axes[0, 0].set_title("不同职称的平均点赞")
    apply_axis_style(axes[0, 0])

    seq_plot = thumb_by_seq.sort_index().head(15)
    axes[0, 1].plot(seq_plot.index.astype(int), seq_plot.values, color=PALETTE["purple"], marker="o", linewidth=2)
    axes[0, 1].set_title("不同回答位次的平均点赞")
    axes[0, 1].set_xlabel("seq")
    apply_axis_style(axes[0, 1])

    axes[1, 0].barh(answer_len_by_dept.index[::-1], answer_len_by_dept.values[::-1], color=PALETTE["cyan"])
    axes[1, 0].set_title("不同科室的平均回答长度")
    apply_axis_style(axes[1, 0])

    axes[1, 1].bar(bounty_group_ans_count.index.astype(str), bounty_group_ans_count.values, color=[PALETTE["blue_soft"], PALETTE["orange"]])
    axes[1, 1].set_title("高悬赏与低悬赏问题的平均回答数")
    apply_axis_style(axes[1, 1])

    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return [figure_path.name]


def main() -> None:
    ensure_directories()
    setup_plot_style()

    question_df = pd.read_parquet(
        QUESTION_PATH,
        columns=[
            "question_id",
            "create_time",
            "gender",
            "age",
            "address",
            "question_desc",
            "tags",
            "bounty",
            "views",
            "ans_count",
        ],
    ).rename(columns={"create_time": "question_create_time"})
    question_df["question_desc"] = question_df["question_desc"].map(clean_text)
    question_df["tags"] = question_df["tags"].map(clean_text)
    question_df["address"] = question_df["address"].map(clean_text)
    question_df["gender"] = question_df["gender"].map(clean_text)
    question_df["question_length"] = question_df["question_desc"].map(text_length)
    question_df["question_month"] = to_month_string(question_df["question_create_time"])

    question_create_time_map = question_df.set_index("question_id")["question_create_time"]
    bounty_map = question_df.set_index("question_id")["bounty"]
    question_actual_count: Counter[int] = Counter()

    doctor_ids: set[str] = set()
    dept_counter: Counter[str] = Counter()
    tech_title_counter: Counter[str] = Counter()
    hospital_counter: Counter[str] = Counter()
    seq_counter: Counter[int] = Counter()
    gender_counter = Counter(value for value in question_df["gender"] if value)
    address_counter = Counter(value for value in question_df["address"] if value)
    tag_counter: Counter[str] = Counter()
    month_answer_counter: Counter[str] = Counter()
    thumb_by_title_sum: defaultdict[str, float] = defaultdict(float)
    thumb_by_title_count: defaultdict[str, int] = defaultdict(int)
    thumb_by_seq_sum: defaultdict[int, float] = defaultdict(float)
    thumb_by_seq_count: defaultdict[int, int] = defaultdict(int)
    answer_len_by_dept_sum: defaultdict[str, float] = defaultdict(float)
    answer_len_by_dept_count: defaultdict[str, int] = defaultdict(int)

    answer_lengths_parts: list[np.ndarray] = []
    thumb_parts: list[np.ndarray] = []
    seq_parts: list[np.ndarray] = []
    response_delay_parts: list[np.ndarray] = []
    answer_text_samples: list[str] = []

    for tags in question_df["tags"]:
        tag_counter.update(split_tags(tags))

    answer_file = pq.ParquetFile(ANSWER_PATH)
    columns = [
        "question_id",
        "user_id",
        "tech_title",
        "dept",
        "hospital",
        "answer",
        "create_time",
        "thumb",
        "help_status",
        "seq",
    ]

    for batch in answer_file.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        batch_df = batch.to_pandas()
        batch_df = batch_df.rename(columns={"create_time": "answer_create_time"})
        batch_df["user_id"] = batch_df["user_id"].map(clean_text)
        batch_df["tech_title"] = batch_df["tech_title"].map(clean_text)
        batch_df["dept"] = batch_df["dept"].map(clean_text)
        batch_df["hospital"] = batch_df["hospital"].map(clean_text)
        batch_df["answer"] = batch_df["answer"].map(clean_text)
        batch_df["answer_length"] = batch_df["answer"].map(text_length)
        batch_df["answer_month"] = to_month_string(batch_df["answer_create_time"])

        batch_df["question_create_time"] = batch_df["question_id"].map(question_create_time_map)
        delay_hours = (
            (batch_df["answer_create_time"] - batch_df["question_create_time"]).dt.total_seconds() / 3600
        )
        delay_hours = delay_hours.where(delay_hours >= 0)

        doctor_ids.update(value for value in batch_df["user_id"] if value)
        dept_counter.update(value for value in batch_df["dept"] if value)
        tech_title_counter.update(value for value in batch_df["tech_title"] if value)
        hospital_counter.update(value for value in batch_df["hospital"] if value)
        seq_counter.update(int(value) for value in batch_df["seq"].dropna().astype(int))
        month_answer_counter.update(batch_df["answer_month"].dropna().astype(str))
        question_actual_count.update(batch_df["question_id"].dropna().astype(int))

        answer_lengths_parts.append(batch_df["answer_length"].to_numpy(dtype=float))
        thumb_parts.append(batch_df["thumb"].fillna(0).to_numpy(dtype=float))
        seq_parts.append(batch_df["seq"].fillna(0).to_numpy(dtype=float))
        response_delay_parts.append(delay_hours.to_numpy(dtype=float))

        for row in batch_df[["tech_title", "thumb"]].itertuples(index=False):
            title, thumb = row
            if title:
                thumb_by_title_sum[title] += float(thumb)
                thumb_by_title_count[title] += 1

        for row in batch_df[["seq", "thumb"]].dropna().itertuples(index=False):
            seq_value, thumb = row
            seq_int = int(seq_value)
            thumb_by_seq_sum[seq_int] += float(thumb)
            thumb_by_seq_count[seq_int] += 1

        for row in batch_df[["dept", "answer_length"]].itertuples(index=False):
            dept, answer_length = row
            if dept:
                answer_len_by_dept_sum[dept] += float(answer_length)
                answer_len_by_dept_count[dept] += 1

        if len(answer_text_samples) < WORD_SAMPLE_LIMIT:
            remain = WORD_SAMPLE_LIMIT - len(answer_text_samples)
            answer_text_samples.extend(batch_df["answer"].head(remain).tolist())

    question_lengths = question_df["question_length"].to_numpy(dtype=float)
    answer_lengths = np.concatenate(answer_lengths_parts) if answer_lengths_parts else np.array([], dtype=float)
    thumb_values = np.concatenate(thumb_parts) if thumb_parts else np.array([], dtype=float)
    seq_values = np.concatenate(seq_parts) if seq_parts else np.array([], dtype=float)
    response_delay_hours = np.concatenate(response_delay_parts) if response_delay_parts else np.array([], dtype=float)
    seq_counts = pd.Series(seq_counter).sort_index()
    answer_count_by_question = pd.Series(question_actual_count).sort_index()
    observed_by_question = answer_count_by_question.rename("observed_answer_count")

    question_df = question_df.merge(observed_by_question, left_on="question_id", right_index=True, how="left")
    question_df["observed_answer_count"] = question_df["observed_answer_count"].fillna(0).astype(int)

    bounty_values = question_df["bounty"].fillna(0)
    bounty_median = float(bounty_values.median())
    if bounty_median <= 0:
        question_df["bounty_group"] = np.where(question_df["bounty"].fillna(0) > 0, "高悬赏", "低悬赏")
        bounty_rule = "bounty>0"
    else:
        question_df["bounty_group"] = np.where(question_df["bounty"].fillna(0) > bounty_median, "高悬赏", "低悬赏")
        bounty_rule = f"bounty>{bounty_median}"

    thumb_by_title = (
        pd.Series({key: thumb_by_title_sum[key] / thumb_by_title_count[key] for key in thumb_by_title_sum})
        .sort_values(ascending=False)
        .head(15)
    )
    thumb_by_seq = pd.Series(
        {key: thumb_by_seq_sum[key] / thumb_by_seq_count[key] for key in thumb_by_seq_sum}
    ).sort_index()
    answer_len_by_dept = (
        pd.Series({key: answer_len_by_dept_sum[key] / answer_len_by_dept_count[key] for key in answer_len_by_dept_sum})
        .sort_values(ascending=False)
        .head(15)
    )
    bounty_group_ans_count = (
        question_df.groupby("bounty_group")["observed_answer_count"].mean().reindex(["低悬赏", "高悬赏"]).dropna()
    )

    dept_top = pd.Series(dept_counter).sort_values(ascending=False).head(15)
    tech_title_top = pd.Series(tech_title_counter).sort_values(ascending=False).head(15)
    hospital_top = pd.Series(hospital_counter).sort_values(ascending=False).head(15)
    gender_counts = pd.Series(gender_counter).sort_values(ascending=False)
    address_top = pd.Series(address_counter).sort_values(ascending=False).head(15)
    tag_top = pd.Series(tag_counter).sort_values(ascending=False).head(15)
    monthly_question_counts = question_df.groupby("question_month").size().sort_index()
    monthly_answer_counts = pd.Series(month_answer_counter).sort_index()

    question_word_counter = tokenize_texts(question_df["question_desc"].tolist(), WORD_SAMPLE_LIMIT)
    answer_word_counter = tokenize_texts(answer_text_samples, WORD_SAMPLE_LIMIT)
    question_word_top = pd.Series(dict(question_word_counter.most_common(20)))
    answer_word_top = pd.Series(dict(answer_word_counter.most_common(20)))

    patient_feature_rows = [
        {
            "field": "gender",
            "missing_count": int((question_df["gender"] == "").sum()),
            "missing_rate": round(float((question_df["gender"] == "").mean()), 4),
            "non_missing_unique_count": int(question_df.loc[question_df["gender"] != "", "gender"].nunique()),
        },
        {
            "field": "age",
            "missing_count": int(question_df["age"].isna().sum()),
            "missing_rate": round(float(question_df["age"].isna().mean()), 4),
            "non_missing_unique_count": int(question_df["age"].nunique(dropna=True)),
        },
        {
            "field": "address",
            "missing_count": int((question_df["address"] == "").sum()),
            "missing_rate": round(float((question_df["address"] == "").mean()), 4),
            "non_missing_unique_count": int(question_df.loc[question_df["address"] != "", "address"].nunique()),
        },
    ]
    patient_feature_df = pd.DataFrame(patient_feature_rows)

    sample_metrics = {
        "question_count": int(question_df["question_id"].nunique()),
        "answer_count": int(len(answer_lengths)),
        "doctor_count": int(len(doctor_ids)),
        "avg_answers_per_question": round(float(answer_count_by_question.mean()), 3),
        "question_share_with_multi_answers": float((answer_count_by_question >= 2).mean()),
        "answer_share_seq_ge_2": float((seq_values >= 2).mean()),
        "question_text_median_length": float(np.nanmedian(question_lengths)),
        "answer_text_median_length": float(np.nanmedian(answer_lengths)),
        "response_delay_median_hours": float(np.nanmedian(response_delay_hours)),
        "response_delay_p95_hours": float(np.nanquantile(response_delay_hours[~np.isnan(response_delay_hours)], 0.95)),
        "thumb_median": float(np.nanmedian(thumb_values)),
        "thumb_p95": float(np.nanquantile(thumb_values, 0.95)),
        "observed_vs_recorded_ans_count_match_rate": float(
            (
                question_df["observed_answer_count"].fillna(0).astype(int)
                == question_df["ans_count"].fillna(-1).astype(int)
            ).mean()
        ),
        "bounty_group_rule": bounty_rule,
    }

    metrics_df = pd.DataFrame(
        [
            {"metric": "question_count", "value": sample_metrics["question_count"]},
            {"metric": "answer_count", "value": sample_metrics["answer_count"]},
            {"metric": "doctor_count", "value": sample_metrics["doctor_count"]},
            {"metric": "avg_answers_per_question", "value": sample_metrics["avg_answers_per_question"]},
            {"metric": "question_share_with_multi_answers", "value": round(sample_metrics["question_share_with_multi_answers"], 4)},
            {"metric": "answer_share_seq_ge_2", "value": round(sample_metrics["answer_share_seq_ge_2"], 4)},
            {"metric": "question_text_median_length", "value": round(sample_metrics["question_text_median_length"], 2)},
            {"metric": "answer_text_median_length", "value": round(sample_metrics["answer_text_median_length"], 2)},
            {"metric": "response_delay_median_hours", "value": round(sample_metrics["response_delay_median_hours"], 2)},
            {"metric": "response_delay_p95_hours", "value": round(sample_metrics["response_delay_p95_hours"], 2)},
            {"metric": "thumb_median", "value": round(sample_metrics["thumb_median"], 2)},
            {"metric": "thumb_p95", "value": round(sample_metrics["thumb_p95"], 2)},
            {
                "metric": "observed_vs_recorded_ans_count_match_rate",
                "value": round(sample_metrics["observed_vs_recorded_ans_count_match_rate"], 4),
            },
            {"metric": "bounty_group_rule", "value": sample_metrics["bounty_group_rule"]},
        ]
    )

    descriptive_relationships_df = pd.DataFrame(
        [
            {"relation": "avg_thumb_by_tech_title", "label": key, "value": round(value, 4)}
            for key, value in thumb_by_title.items()
        ]
        + [
            {"relation": "avg_thumb_by_seq", "label": str(key), "value": round(value, 4)}
            for key, value in thumb_by_seq.head(15).items()
        ]
        + [
            {"relation": "avg_answer_length_by_dept", "label": key, "value": round(value, 4)}
            for key, value in answer_len_by_dept.items()
        ]
        + [
            {"relation": "avg_ans_count_by_bounty_group", "label": key, "value": round(value, 4)}
            for key, value in bounty_group_ans_count.items()
        ]
    )

    word_frequency_df = pd.DataFrame(
        [{"source": "question_desc", "token": key, "count": value} for key, value in question_word_counter.most_common(20)]
        + [{"source": "answer", "token": key, "count": value} for key, value in answer_word_counter.most_common(20)]
        + [{"source": "tags", "token": key, "count": value} for key, value in tag_counter.most_common(20)]
    )

    save_dataframe(metrics_df, SAMPLE_METRICS_PATH)
    save_dataframe(patient_feature_df, PATIENT_FEATURES_PATH)
    save_dataframe(descriptive_relationships_df, DESCRIPTIVE_RELATIONSHIPS_PATH)
    save_dataframe(word_frequency_df, WORD_FREQUENCY_PATH)
    save_dataframe(pd.DataFrame({"dept": dept_top.index, "count": dept_top.values}), TABLES_DIR / "02_dept_distribution.csv")
    save_dataframe(pd.DataFrame({"tech_title": tech_title_top.index, "count": tech_title_top.values}), TABLES_DIR / "02_tech_title_distribution.csv")
    save_dataframe(pd.DataFrame({"hospital": hospital_top.index, "count": hospital_top.values}), TABLES_DIR / "02_hospital_concentration.csv")
    save_dataframe(pd.DataFrame({"seq": seq_counts.index, "count": seq_counts.values}), TABLES_DIR / "02_seq_distribution.csv")
    save_dataframe(
        pd.DataFrame({"answers_per_question": answer_count_by_question.value_counts().sort_index().index, "question_count": answer_count_by_question.value_counts().sort_index().values}),
        TABLES_DIR / "02_answers_per_question_distribution.csv",
    )

    figure_files: list[str] = []
    figure_files.extend(plot_sample_and_thread(sample_metrics, seq_counts, answer_count_by_question, thumb_values))
    figure_files.extend(plot_structure_distributions(dept_top, tech_title_top, hospital_top))
    figure_files.extend(plot_text_and_time(question_lengths, answer_lengths, monthly_question_counts, monthly_answer_counts, response_delay_hours))
    figure_files.extend(plot_patient_and_tags(gender_counts, question_df["age"].dropna().to_numpy(dtype=float), address_top, tag_top))
    figure_files.extend(plot_word_frequencies(question_word_top, answer_word_top))
    figure_files.extend(plot_descriptive_relationships(thumb_by_title, thumb_by_seq, answer_len_by_dept, bounty_group_ans_count))

    summary = {
        "run_ts_utc": utc_now_iso(),
        "scope": "answer_question_only",
        "question_file": str(QUESTION_PATH),
        "answer_file": str(ANSWER_PATH),
        "sample_metrics": sample_metrics,
        "outputs": {
            "summary_json": str(SUMMARY_PATH),
            "tables": [
                str(SAMPLE_METRICS_PATH),
                str(PATIENT_FEATURES_PATH),
                str(DESCRIPTIVE_RELATIONSHIPS_PATH),
                str(WORD_FREQUENCY_PATH),
            ],
            "figures": [str(FIGURES_DIR / file_name) for file_name in figure_files],
        },
    }

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    append_cleaning_log(
        [
            {
                "run_ts_utc": summary["run_ts_utc"],
                "step": "01b_eda_overview",
                "action": "generate_exploratory_statistics_and_figures",
                "status": "completed",
                "target_path": str(SUMMARY_PATH),
                "details": f"scope={summary['scope']}; figures={len(figure_files)}; bounty_group_rule={bounty_rule}",
            }
        ]
    )

    print(f"EDA summary written to {SUMMARY_PATH}")
    print(f"Figures generated: {len(figure_files)}")
    print(f"Tables generated in {TABLES_DIR}")


if __name__ == "__main__":
    main()
