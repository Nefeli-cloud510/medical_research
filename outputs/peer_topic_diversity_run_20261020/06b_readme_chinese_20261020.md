# 回答级同行知识多样性指标（A 方案句子主题基线）

运行日期：2026-10-20；运行目录：D:\medical_research\outputs\peer_topic_diversity_run_20261020

## 口径

- 输入：100 个问题的 859 条医生回答；A 句子级主题有效 K=33，保留全部有效主题，含模板簇，不做合并或删除。
- 严格前序：同一 question_id 下 answer_create_time 严格早于焦点回答；同时间戳批次先计算、再整体加入前序。
- 拥挤度：prior_answer_count=|P_ij|，prior_total_char_count=sum(冻结字符数 over P_ij)。
- 主题分布：回答 r 在有效主题 k 的句子占比；-1 噪声句不进分子分母。
- 多样性：先对有效前序回答主题分布等权平均，再计算自然对数 Shannon 熵及主题广度；不做前序各自熵的平均。

## 主要结果

- 输出行数：859；问题数：100；event_id 唯一：True。
- 前序回答数：min=0.0，median=4.0，mean=5.232829，max=24.0。
- 前序总字符数：mean=453.055879，max=2448.0。
- 主题熵：有效=740，缺失=119；mean=0.741341，min=0.0，max=1.929032。
- 主题广度：有效=740；median=3.0，max=11.0。
- 前序覆盖：有前序的回答为 759 行，其中有有效主题分布的回答为 740 行；mean coverage=0.925868。
- 缺失原因：[{"reason": "no_strict_prior", "answer_count": 100}, {"reason": "no_valid_prior_topic_distribution", "answer_count": 19}]。

## 验证

- 全部 859 行均用独立直接筛选方式重查前序数量、字符数、主题覆盖、熵与广度；mismatch=0。
- 与已有 flow_run100_knowledge_metrics_bge.csv 按 event_id 一对一连接，行数不变；count/char 匹配均为 859。
- 分布非负、有效分布和约等于 1；熵在 [0, ln(K)] 内。

## 范围说明

- 前序信息集合限定于当前 flow_run 清洗后的 859 条回答；该样本来自核心职称与主要医院等级过滤后的回答库，未把被过滤掉的平台回答补回。
- 本版为探索性基线，不用于正式回归结论；主题熵暂不解释为已经验证的纯医学知识多样性。
