from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT_DIR / "data" / "analysis"
OUTPUTS_DIR = ROOT_DIR / "outputs"
RUN_ROOT = OUTPUTS_DIR / "topic_ab_run_20261020"
AUDIT_DIR = OUTPUTS_DIR / "audit"
TABLE_DIR = OUTPUTS_DIR / "tables"
USER_OUTPUTS = Path(r"C:\Users\admin\Documents\Codex\2026-10-19\co\outputs")

ANSWER_CSV = ANALYSIS_DIR / "flow_run100_answer_text_for_topic_clean.csv"
SENTENCE_CSV = ANALYSIS_DIR / "flow_run100_sentence_text_for_topic_clean.csv"

ID_STR_COLUMNS = ["question_id", "event_id", "user_id"]

# (candidate_name_zh, one_line_definition, mixed_content_flag, template_dominant_flag)
A_CANDIDATES = {
    0: ("皮肤过敏与湿疹护理", "围绕湿疹、丘疹性荨麻疹、过敏诱因与外用药物护理。", False, False),
    1: ("膝关节与关节损伤的运动康复", "涉及膝关节软骨、半月板、韧带损伤、注射治疗和运动后关节症状。", False, False),
    2: ("泌尿生殖与包皮前列腺问题", "以包皮、阴茎、前列腺、尿常规和泌尿系检查为主的一组问题。", False, False),
    3: ("慢性胃炎溃疡与幽门螺杆菌治疗", "胃炎溃疡的抑酸治疗、根除幽门螺杆菌和胃镜评估。", False, False),
    4: ("肿瘤胸膜转移与胸腔化疗", "肿瘤晚期胸膜转移、胸腔灌注化疗和放疗相关。", False, False),
    5: ("胆道闭锁与小儿外科手术", "先天性胆道闭锁、胆汁淤积和手术/肝移植评估。", False, False),
    6: ("用药剂量与肾功能注意事项", "服药方式、药物剂量、肾损伤风险和就诊提示。", False, True),
    7: ("脑梗死与脑血管介入评估", "脑梗死、脑血管狭窄闭塞、造影和介入/康复治疗。", False, False),
    8: ("诊断明确度与靶向治疗决策", "混杂诊断解释、靶向治疗和后续治疗建议。", True, False),
    9: ("早孕期出血与保胎", "早孕期出血、先兆流产、孕酮和保胎评估。", False, False),
    10: ("月经推迟与孕激素用药", "达芙通等孕激素、月经推迟、子宫内膜与HCG检查。", False, False),
    11: ("慢性咳嗽与气道疾病鉴别", "慢性咳嗽、咳嗽变异性哮喘等气道疾病鉴别。", False, False),
    12: ("检查必要性评估与安抚", "检查是否必要的判断和“不用担心”类安抚。", False, False),
    13: ("祝福与早日康复模板句", "以祝早日康复等结尾语为主，缺少实质医学知识。", False, True),
    14: ("HPV感染与宫颈癌筛查", "HPV分型、TCT和宫颈癌筛查随访。", False, False),
    15: ("血管瘤观察与手术时机", "婴幼儿血管瘤性质、复查和手术时机。", False, False),
    16: ("清淡饮食与忌口建议", "清淡饮食、忌辛辣刺激等通用饮食建议，模板化较高。", False, True),
    17: ("心电图早搏与动态心电图", "心电检查、早搏和进一步心脏检查。", False, False),
    18: ("休息睡眠与生活方式建议", "注意休息、睡眠、戒烟戒酒等通用生活方式建议，模板化。", False, True),
    19: ("儿童淋巴结肿大与观察", "儿童颈部/耳后淋巴结肿大、彩超和观察建议。", False, False),
    20: ("医院就诊与挂号指引", "医院预约挂号、面诊建议和互联网诊疗免责声明，混有少量诊疗提示。", True, True),
    21: ("儿童发热与呼吸道感染", "儿童发热、呼吸道感染、血常规和肺炎排查。", False, False),
    22: ("根据描述供参考的开场模板", "固定开场“根据您描述的情况，给您意见供参考”，不含实质知识。", False, True),
    23: ("希望对您有所帮助的结尾模板", "结束语“希望对您有所帮助”，不含实质知识。", False, True),
    24: ("中耳炎外耳道与听力手术", "中耳炎、耳内镜、外耳道肿物和听力手术。", False, False),
    25: ("儿童身高发育与骨龄评估", "儿童骨龄、青春期早发育、遗传身高预测和营养建议。", False, False),
    26: ("追问症状的信息收集句", "“有什么症状吗”等追问或信息收集句。", False, True),
    27: ("宫颈活检后出血与止血", "宫颈活检后出血量、止血和复查建议。", False, False),
    28: ("减肥药物安全与体重管理", "减肥药副作用、饮食运动与体重控制。", False, False),
    29: ("孕期药物对胚胎的全或无影响", "早孕期药物对胚胎的全或无影响与流产风险解释。", False, False),
    30: ("糖尿病诊断与血糖控制", "血糖标准、糖尿病诊断和胰岛素控制。", False, False),
    31: ("低蛋白血症与下肢水肿", "下肢水肿原因、低蛋白血症和肝肾功能评估。", False, False),
    32: ("智齿拔除与口腔颌面外科", "智齿拔除、牙片评估和术后咀嚼影响。", False, False),
}

B_CANDIDATES = {
    0: ("儿童皮炎湿疹与皮肤护理", "以湿疹、荨麻疹等皮肤问题和外用药护理为主。", False, False),
    1: ("膝关节骨关节炎与保守治疗", "膝关节骨关节炎、软骨损伤和保守治疗建议。", False, False),
    2: ("儿童身高喂养与发育评估", "儿童身高体重、喂养营养、发育和就医建议，内容较宽。", True, False),
    3: ("慢性胃炎溃疡与幽门螺杆菌治疗", "胃炎溃疡、幽门螺杆菌和消化道用药治疗。", False, False),
    4: ("早孕期用药与胚胎影响", "早孕期用药、胚胎影响、流产风险和孕期检查。", False, False),
    5: ("泌尿系统与肾功能检查", "尿常规、尿蛋白、泌尿系和水肿原因检查，内容较混杂。", True, False),
    6: ("慢性咳嗽与气道疾病鉴别", "慢性咳嗽、鼻后滴漏、咳嗽变异性哮喘等气道鉴别。", False, False),
    7: ("肺癌胸膜转移与抗肿瘤治疗", "肺癌晚期、胸膜转移、化疗免疫治疗和生存期讨论。", False, False),
    8: ("月经孕激素与宫颈检查", "月经推迟、孕激素、子宫内膜、HPV和宫颈活检，内容较混杂。", True, False),
    9: ("挂号预约与就诊指引", "挂号、预约、互联网诊疗规范及少量药物信息，模板化。", True, True),
    10: ("阴茎勃起与包皮外阴问题", "阴茎勃起、包皮、假性湿疣等外生殖器问题。", False, False),
    11: ("婴幼儿血管瘤与手术时机", "婴幼儿血管瘤性质、复查和手术时机。", False, False),
    12: ("儿童淋巴结肿大与彩超观察", "儿童淋巴结肿大、彩超观察和门诊评估。", False, False),
    13: ("糖尿病减肥与体重管理", "糖尿病、血糖、胰岛素、减肥和体重管理，内容较混杂。", True, False),
    14: ("心电图早搏与心脏检查", "心电图、早搏、动态心电图和心脏进一步检查。", False, False),
    15: ("脑梗死与脑血管介入评估", "脑梗死、脑血管闭塞、造影和介入治疗评估。", False, False),
}

RESIDUAL_PATTERNS = [
    ("问候开场", re.compile(r"^(您好|你好|很高兴|感谢咨询|谢谢咨询)")),
    ("祝福结尾", re.compile(r"祝您|祝你|早日康复|生活愉快|身体健康|安康")),
    ("诊疗免责模板", re.compile(r"仅供参考|不代表.*诊疗|以面诊为准|最终(诊疗|诊断)结果")),
    ("就诊转介模板", re.compile(r"建议(您)?到|建议(您)?去|前往医院|三甲医院|正规医院")),
    ("挂号预约模板", re.compile(r"预约|挂号")),
    ("安抚常用句", re.compile(r"不用(太)?担心|不要担心|没什么大问题|不用紧张")),
    ("生活方式模板", re.compile(r"清淡饮食|忌.*辛辣|注意休息|保证(充足)?睡眠|戒烟戒酒")),
]


def load_answer_df() -> pd.DataFrame:
    df = pd.read_csv(ANSWER_CSV, encoding="utf-8-sig", dtype={column: str for column in ID_STR_COLUMNS})
    df["answer_create_time"] = pd.to_datetime(df["answer_create_time"])
    df["time_seq"] = pd.to_numeric(df["time_seq"], errors="coerce").astype("Int64")
    df["seq"] = pd.to_numeric(df["seq"], errors="coerce").astype("Int64")
    return df.sort_values(["question_id", "time_seq", "answer_create_time", "seq"], kind="stable").reset_index(drop=True)


def load_sentence_df(question_ids: set[str]) -> pd.DataFrame:
    df = pd.read_csv(SENTENCE_CSV, encoding="utf-8-sig", dtype={column: str for column in ID_STR_COLUMNS})
    df["question_id"] = df["question_id"].astype(str)
    df["time_seq"] = pd.to_numeric(df["time_seq"], errors="coerce").astype("Int64")
    df["seq"] = pd.to_numeric(df["seq"], errors="coerce").astype("Int64")
    df = df.loc[df["question_id"].isin(question_ids)].copy()
    if "drop_for_topic_cleaning" in df.columns:
        flag = df["drop_for_topic_cleaning"].fillna(False).astype(str).str.strip().str.lower()
        flag = flag.map({"true": True, "false": False, "1": True, "0": False})
        df["drop_for_topic_cleaning"] = flag.fillna(False).astype(bool)
        df = df.loc[~df["drop_for_topic_cleaning"]].copy()
    return df.reset_index(drop=True)


def build_sample_stats() -> None:
    answer_df = load_answer_df()
    sentence_df = load_sentence_df(set(answer_df["question_id"].tolist()))
    answer_df["raw_answer_len"] = answer_df["focal_answer_text_raw"].fillna("").astype(str).str.len()
    answer_df["clean_answer_len"] = answer_df["answer_text_topic_clean"].fillna("").astype(str).str.len()
    answer_df["sentence_count"] = answer_df["kept_sentence_count"] + answer_df["dropped_sentence_count"]

    question_stats_rows = []
    for question_id, group in answer_df.groupby("question_id", sort=False):
        question_stats_rows.append(
            {
                "question_id": str(question_id),
                "answer_count": int(len(group)),
                "clean_answer_count": int(group["has_clean_topic_text"].astype(bool).sum()),
                "sentence_count": int(group["sentence_count"].sum()),
                "kept_sentence_count": int(group["kept_sentence_count"].sum()),
                "dropped_sentence_count": int(group["dropped_sentence_count"].sum()),
                "raw_answer_len_min": int(group["raw_answer_len"].min()),
                "raw_answer_len_median": int(group["raw_answer_len"].median()),
                "raw_answer_len_max": int(group["raw_answer_len"].max()),
                "raw_answer_len_mean": round(float(group["raw_answer_len"].mean()), 2),
            }
        )
    question_stats = pd.DataFrame.from_records(question_stats_rows)
    question_stats.to_csv(TABLE_DIR / "flow_run100_100_sample_stats_per_question.csv", index=False, encoding="utf-8-sig")
    answer_df.to_csv(TABLE_DIR / "flow_run100_100_sample_stats_per_answer.csv", index=False, encoding="utf-8-sig")

    summary = {
        "scope": "flow_run100_100_sample_statistics",
        "question_count": int(answer_df["question_id"].nunique()),
        "answer_count": int(len(answer_df)),
        "answer_count_per_question": {
            "min": int(question_stats["answer_count"].min()),
            "median": int(question_stats["answer_count"].median()),
            "max": int(question_stats["answer_count"].max()),
        },
        "sentence_count_total": int(sentence_df["sentence_id"].nunique() if "sentence_id" in sentence_df.columns else len(sentence_df)),
        "sentence_count_kept": int(answer_df["kept_sentence_count"].sum()),
        "sentence_count_dropped": int(answer_df["dropped_sentence_count"].sum()),
        "answer_clean_count": int(answer_df["has_clean_topic_text"].astype(bool).sum()),
        "answer_without_clean_count": int((~answer_df["has_clean_topic_text"].astype(bool)).sum()),
        "raw_answer_length": {
            "min": int(answer_df["raw_answer_len"].min()),
            "median": int(answer_df["raw_answer_len"].median()),
            "max": int(answer_df["raw_answer_len"].max()),
        },
        "clean_answer_length": {
            "min": int(answer_df["clean_answer_len"].min()),
            "median": int(answer_df["clean_answer_len"].median()),
            "max": int(answer_df["clean_answer_len"].max()),
        },
    }
    (AUDIT_DIR / "20k_flow_run100_100_sample_stats_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("sample stats written")


def _ensure_candidate_row(unit: str, topic_id: int) -> dict[str, object]:
    candidates = A_CANDIDATES if unit == "sentence" else B_CANDIDATES
    item = candidates.get(int(topic_id))
    if item is None:
        return {
            "candidate_name_zh": f"待人工确认-主题{topic_id}",
            "one_line_definition": "待人工补写",
            "manual_flag": "待人工确认",
            "mixed_medical_content_notes": "待人工判断",
            "template_dominant_notes": "待人工判断",
        }
    return {
        "candidate_name_zh": item[0],
        "one_line_definition": item[1],
        "manual_flag": "待人工确认",
        "mixed_medical_content_notes": "是，混杂多个医学内容，待确认" if item[2] else "暂未见明显混杂，待确认",
        "template_dominant_notes": "是，模板句为主，待确认" if item[3] else "否，以实质内容为主",
    }


def fill_audit_tables() -> None:
    for unit, dir_name in [("sentence", "A_sentence_no_reduce"), ("answer", "B_answer_no_reduce")]:
        source = RUN_ROOT / dir_name / f"{unit}_topic_audit.csv"
        df = pd.read_csv(source, encoding="utf-8-sig")
        rows = []
        for _, row in df.iterrows():
            filled = _ensure_candidate_row(unit, int(row["topic_id"]))
            item = row.to_dict()
            item.update(filled)
            rows.append(item)
        out_df = pd.DataFrame.from_records(rows)
        project_path = RUN_ROOT / dir_name / f"{unit}_topic_audit_zh_20261020.csv"
        audit_path = AUDIT_DIR / f"中文主题审计表_20261020_{unit}.csv"
        out_df.to_csv(project_path, index=False, encoding="utf-8-sig")
        out_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print("zh audit tables written")


def build_manual_review_samples() -> None:
    answer_df = load_answer_df()
    shuffled = answer_df.sample(frac=1.0, random_state=20261020)
    shuffled["raw_len"] = shuffled["focal_answer_text_raw"].fillna("").astype(str).str.len()
    selected_rows: list[pd.Series] = []
    per_question_counts: dict[str, int] = {}
    for _, row in shuffled.iterrows():
        question_id = str(row["question_id"])
        if per_question_counts.get(question_id, 0) >= 3:
            continue
        selected_rows.append(row)
        per_question_counts[question_id] = per_question_counts.get(question_id, 0) + 1
        if len(selected_rows) >= 45:
            break
    sample = pd.DataFrame(selected_rows)
    if sample.empty:
        raise RuntimeError("No manual review samples selected.")
    sample = sample.sort_values(["question_id", "time_seq", "answer_create_time", "seq"]).reset_index(drop=True)

    a_stats = pd.read_csv(
        RUN_ROOT / "A_sentence_no_reduce" / "answer_topic_stats_sentence_level.csv",
        encoding="utf-8-sig",
        dtype={"event_id": str, "question_id": str},
    )
    b_labels = pd.read_csv(
        RUN_ROOT / "B_answer_no_reduce" / "answer_topic_labels_answer_level.csv",
        encoding="utf-8-sig",
        dtype={"event_id": str, "question_id": str},
    )
    sample = sample.merge(
        a_stats[["event_id", "dominant_topic_id", "topic_coverage"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    sample = sample.merge(
        b_labels[["event_id", "topic_id"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    sample["A_主导主题"] = sample["dominant_topic_id"].map(
        lambda x: (
            _ensure_candidate_row("sentence", int(x))["candidate_name_zh"] if pd.notna(x) else "无有效主题"
        )
    )
    sample["B_回答主题"] = sample.apply(
        lambda row: (
            _ensure_candidate_row("answer", int(row["topic_id"]))["candidate_name_zh"]
            if pd.notna(row["topic_id"])
            else ""
        ),
        axis=1,
    )
    sample["sample_id"] = [f"manual_{index + 1:02d}" for index in range(len(sample))]
    sample["人工判断_A_主题是否合理"] = ""
    sample["人工判断_B_主题是否合理"] = ""
    sample["人工判断_主题是否符合原文(不看标签先判断)"] = ""
    sample["备注"] = ""

    renamed = sample.rename(
        columns={
            "raw_len": "回答长度(字符)",
            "focal_answer_text_raw": "医生回答原文",
            "question_text": "患者问题原文",
        }
    )
    pure_columns = [
        "sample_id",
        "question_id",
        "event_id",
        "answer_create_time",
        "患者问题原文",
        "医生回答原文",
        "回答长度(字符)",
        "人工判断_主题是否符合原文(不看标签先判断)",
        "备注",
    ]
    pure_out = renamed[pure_columns]
    pure_out.to_csv(AUDIT_DIR / "人工核验样本_纯原文_20261020.csv", index=False, encoding="utf-8-sig")

    with_label = sample.rename(
        columns={
            "raw_len": "回答长度(字符)",
            "focal_answer_text_raw": "医生回答原文",
            "question_text": "患者问题原文",
            "question_id": "question_id(字符串)",
            "event_id": "event_id(字符串)",
            "dominant_topic_id": "A_句子级_主导主题ID",
            "topic_id": "B_回答级_主题ID",
            "topic_coverage": "A_回答覆盖主题数(句子级)",
        }
    )
    with_label.to_csv(AUDIT_DIR / "人工核验样本_含主题对照_20261020.csv", index=False, encoding="utf-8-sig")
    print("manual review samples written", len(sample))


def build_comparison() -> None:
    rows = []
    for unit, dir_name in [("sentence", "A_sentence_no_reduce"), ("answer", "B_answer_no_reduce")]:
        fit = json.loads((RUN_ROOT / dir_name / "fit_summary.json").read_text(encoding="utf-8"))
        audit = pd.read_csv(RUN_ROOT / dir_name / f"{unit}_topic_audit.csv", encoding="utf-8-sig")
        weighted_template = float(
            (audit["topic_doc_count"] * audit["template_share_heuristic"]).sum() / audit["topic_doc_count"].sum()
        )
        weighted_medical = float(
            (audit["topic_doc_count"] * audit["medical_marker_share_heuristic"]).sum() / audit["topic_doc_count"].sum()
        )
        empty_keywords = int((audit["keywords_zh_top20"].fillna("").astype(str).str.strip() == "").sum())
        rows.append(
            {
                "unit": unit,
                "display_name": fit["unit_display_name"],
                "rows": fit["counts"]["rows_total"],
                "valid_topic_count": fit["counts"]["valid_topic_count"],
                "initial_topic_count_including_noise": fit["algorithm"]["initial_topic_count_including_noise"],
                "noise_rows": fit["counts"]["noise_rows"],
                "noise_rate": fit["counts"]["noise_rate"],
                "max_topic_id": fit["counts"]["max_topic_id"],
                "max_topic_count": fit["counts"]["max_topic_count"],
                "max_topic_share": fit["counts"]["max_topic_share"],
                "answer_level_coverage": fit["counts"]["answer_level_coverage"],
                "weighted_template_share": round(weighted_template, 4),
                "weighted_medical_marker_share": round(weighted_medical, 4),
                "empty_keyword_topic_count": empty_keywords,
                "language": "multilingual",
                "vectorizer_min_df": fit["algorithm"]["vectorizer_min_df"],
                "nr_topics_requested": None,
                "reduction_applied": False,
            }
        )
    comparison_df = pd.DataFrame.from_records(rows)
    comparison_df.to_csv(RUN_ROOT / "topic_ab_comparison_summary.csv", index=False, encoding="utf-8-sig")
    print("comparison written")


def build_clean_v1_residual_audit() -> None:
    answer_df = load_answer_df()
    question_ids = set(answer_df["question_id"].tolist())
    sentence_df = load_sentence_df(question_ids)
    answer_texts = answer_df["answer_text_topic_clean"].fillna("").astype(str).tolist()
    sentence_texts = sentence_df["sentence_text_topic_clean"].fillna("").astype(str).tolist()
    rows = []
    for label, pattern in RESIDUAL_PATTERNS:
        sentence_hits = [text for text in sentence_texts if pattern.search(text)]
        answer_hits = [text for text in answer_texts if pattern.search(text)]
        rows.append(
            {
                "pattern_category": label,
                "pattern": pattern.pattern,
                "sentence_count": len(sentence_hits),
                "sentence_share": round(float(len(sentence_hits) / len(sentence_texts)), 6) if sentence_texts else 0,
                "answer_count": len(answer_hits),
                "answer_share": round(float(len(answer_hits) / len(answer_texts)), 6) if answer_texts else 0,
                "sentence_examples": "\n".join(sentence_hits[:3]),
                "answer_examples": "\n".join(answer_hits[:3]),
            }
        )
    df = pd.DataFrame.from_records(rows)
    df.to_csv(AUDIT_DIR / "clean_v1_residual_template_audit_20261020.csv", index=False, encoding="utf-8-sig")
    print("clean v1 residual audit written")


def copy_user_outputs() -> None:
    USER_OUTPUTS.mkdir(parents=True, exist_ok=True)
    for path in AUDIT_DIR.glob("*.csv"):
        target = USER_OUTPUTS / path.name
        target.write_bytes(path.read_bytes())
    for path in TABLE_DIR.glob("flow_run100_100_sample_stats*.csv"):
        target = USER_OUTPUTS / path.name
        target.write_bytes(path.read_bytes())
    for name in ["中文主题审计表_20261020_sentence.csv", "中文主题审计表_20261020_answer.csv"]:
        source = AUDIT_DIR / name
        if source.exists():
            (USER_OUTPUTS / name).write_bytes(source.read_bytes())
    print("user outputs copied")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    build_sample_stats()
    fill_audit_tables()
    build_manual_review_samples()
    build_comparison()
    build_clean_v1_residual_audit()
    copy_user_outputs()


if __name__ == "__main__":
    main()
