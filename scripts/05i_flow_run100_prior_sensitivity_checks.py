from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
OUTPUT_TABLE_DIR = ROOT_DIR / "outputs" / "tables"
OUTPUT_AUDIT_DIR = ROOT_DIR / "outputs" / "audit"
METADATA_DIR = ROOT_DIR / "metadata"

INPUT_PATH = ANALYSIS_DIR / "flow_run100_knowledge_metrics_bge.parquet"
OUTPUT_XLSX_PATH = OUTPUT_TABLE_DIR / "20d_flow_run100_prior_sensitivity_checks.xlsx"
OUTPUT_JSON_PATH = OUTPUT_AUDIT_DIR / "20d_flow_run100_prior_sensitivity_checks_summary.json"
CLEANING_LOG_PATH = METADATA_DIR / "cleaning_log.csv"

BINS = [0, 1, 2, 4, 9, 19, 24]
LABELS = ["1", "2", "3-4", "5-9", "10-19", "20-24"]
SMALL_SAMPLE_LABEL = "样本较少，仅作观察"

CLEANING_LOG_COLUMNS = [
    "run_ts_utc",
    "step",
    "action",
    "status",
    "target_path",
    "details",
]

METRIC_CONFIG = [
    ("knowledge_differentiation_bge_centroid", "centroid"),
    ("knowledge_differentiation_bge_nearest_prior", "nearest"),
    ("knowledge_differentiation_bge_sentence_mean", "sentence"),
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


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.loc[df["peer_crowding_prior_answer_count"] > 0].copy()
    work_df["prior_group"] = pd.cut(
        work_df["peer_crowding_prior_answer_count"],
        bins=BINS,
        labels=LABELS,
    )

    summary = (
        work_df.groupby("prior_group", observed=True)
        .agg(
            n=("event_id", "count"),
            centroid_mean=("knowledge_differentiation_bge_centroid", "mean"),
            centroid_median=("knowledge_differentiation_bge_centroid", "median"),
            nearest_mean=("knowledge_differentiation_bge_nearest_prior", "mean"),
            nearest_median=("knowledge_differentiation_bge_nearest_prior", "median"),
            sentence_mean=("knowledge_differentiation_bge_sentence_mean", "mean"),
            sentence_median=("knowledge_differentiation_bge_sentence_mean", "median"),
        )
        .reset_index()
    )
    summary["note"] = ""
    summary.loc[summary["prior_group"].astype(str) == "20-24", "note"] = SMALL_SAMPLE_LABEL
    return summary


def build_correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_col, metric_name in METRIC_CONFIG:
        metric_df = df.loc[
            (df["peer_crowding_prior_answer_count"] > 0) & df[metric_col].notna(),
            ["peer_crowding_prior_answer_count", metric_col],
        ].copy()
        x = metric_df["peer_crowding_prior_answer_count"]
        y = metric_df[metric_col]
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_rho, spearman_p = spearmanr(x, y)
        rows.append(
            {
                "metric": metric_name,
                "metric_column": metric_col,
                "n": int(len(metric_df)),
                "pearson_r": float(pearson_r),
                "pearson_pvalue": float(pearson_p),
                "spearman_rho": float(spearman_rho),
                "spearman_pvalue": float(spearman_p),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH).copy()
    group_summary = build_group_summary(df)
    correlation_summary = build_correlation_summary(df)

    with pd.ExcelWriter(OUTPUT_XLSX_PATH, engine="openpyxl") as writer:
        group_summary.to_excel(writer, sheet_name="group_summary", index=False)
        correlation_summary.to_excel(writer, sheet_name="correlations", index=False)

    summary_json = {
        "run_ts_utc": utc_now_iso(),
        "input_path": str(INPUT_PATH),
        "output_xlsx_path": str(OUTPUT_XLSX_PATH),
        "group_summary_rows": int(len(group_summary)),
        "correlation_rows": int(len(correlation_summary)),
        "small_sample_note_group": "20-24",
        "group_summary_preview": group_summary.to_dict(orient="records"),
        "correlation_summary": correlation_summary.to_dict(orient="records"),
    }
    OUTPUT_JSON_PATH.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

    append_cleaning_log(
        [
            {
                "run_ts_utc": summary_json["run_ts_utc"],
                "step": "05i_flow_run100_prior_sensitivity_checks",
                "action": "write_prior_group_summary_and_correlations",
                "status": "completed",
                "target_path": str(OUTPUT_XLSX_PATH),
                "details": (
                    f"group_rows={len(group_summary)}; correlation_rows={len(correlation_summary)}; "
                    f"small_sample_group=20-24"
                ),
            },
            {
                "run_ts_utc": summary_json["run_ts_utc"],
                "step": "05i_flow_run100_prior_sensitivity_checks",
                "action": "write_prior_sensitivity_summary_json",
                "status": "completed",
                "target_path": str(OUTPUT_JSON_PATH),
                "details": (
                    f"input={INPUT_PATH.name}; output={OUTPUT_XLSX_PATH.name}"
                ),
            },
        ]
    )

    print(f"Workbook written to {OUTPUT_XLSX_PATH}")
    print(f"Summary written to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
