# weiyi_project AI交接说明

## 1. 文档目的

这份文档用于把当前 `weiyi_project` 的数据工程与分析准备工作，完整移交给下一个 AI。

目标是让接手者无需重新摸索，即可直接理解：

- 当前项目做到哪一步；
- 每个关键数据版本分别干什么；
- 哪个文件是“基础回答库”、哪个是“分析母表”、哪个是“主回归样本”；
- 哪些字段应该继续使用，哪些字段只是历史审计遗留；
- 接下来文本清洗、前序回答模板构造、文本特征计算应该从哪里接着做。


## 2. 项目背景与研究口径

### 2.1 研究对象

当前项目仅使用微医平台两张原始表：

- `data/raw/answer.parquet`
- `data/raw/question.parquet`

已明确不再使用 `hdf_*` 系列文件作为当前主流程数据源。

### 2.2 当前样本约束

目前主样本已经做过两层核心筛选：

#### 职称筛选

只保留五类核心临床医师职称：

- `副主任医师`
- `主治医师`
- `主任医师`
- `住院医师`
- `医师`

#### 医院等级筛选

在主回归分析样本中，只保留五类医院等级：

- `一级甲等`
- `二级乙等`
- `三级乙等`
- `三级甲等`
- `二级甲等`


## 3. 项目硬约束

接手 AI 必须遵守以下规则：

- `data/raw/` 原始数据绝对只读，不允许覆盖、回写或删除。
- 所有正式处理应通过 `scripts/` 下编号脚本完成。
- 所有新版本输出应写入 `data/staging/`、`data/analysis/`、`outputs/` 等下游目录。
- 每一步重要处理必须在 `metadata/cleaning_log.csv` 留痕。
- 当前研究的主顺序口径是按真实时间重排后的 `time_seq`，不是平台原始 `seq`。


## 4. 目录与当前关键文件

### 4.1 原始数据

- `data/raw/answer.parquet`
- `data/raw/question.parquet`
- `data/raw/wy_hospital.parquet`

### 4.2 当前最重要的分析数据版本

#### A. 基础回答库

- 文件：
  `data/analysis/base_answer_library_core_physician_round2.parquet`
- 用途：
  保存所有可追溯核心医师回答事件，用于复现、回答定位、代理事件键、重复抓取核验、模板回答识别。
- 特点：
  - 保留全部核心医师回答；
  - 不按医院等级 `major5` 再次删样本；
  - 已建立代理事件键；
  - 已识别模板文本复用。

#### B. 全量核心医师分析母表

- 文件：
  `data/analysis/focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_filled_round2.parquet`
- 用途：
  当前最完整的核心医师分析母表；若后续需要重新派生其他研究样本，应从这里开始。
- 特点：
  - 已删除时间异常样本；
  - 已筛到五类核心医师；
  - 医院等级已做两轮补填；
  - 仍保留非 `major5` 的医院等级，不是最终主回归样本。

#### C. 主回归分析库

- 文件：
  `data/analysis/focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_major5.parquet`
- 用途：
  当前主回归分析样本。
- 特点：
  - 五类核心职称；
  - 五类主要医院等级；
  - 适合主回归、主描述统计。

#### D. 文本清洗准备库

- 文件：
  `data/analysis/major5_text_cleaning_prep.parquet`
- 用途：
  在 `major5` 基础上，为文本清洗与实际计算做准备。
- 特点：
  - 已确认问答字段已拼接完成；
  - 已补 `question_id × user_id` 统计字段；
  - 已确认当前样本中不存在“同一医生在同一问题下多次回答”。

#### E. 当前推荐继续工作的版本

- 文件：
  `data/analysis/major5_text_cleaning_prep_seq_clean.parquet`
- 用途：
  当前最推荐继续往下做文本清洗、前序回答模板构造、文本特征计算的版本。
- 特点：
  - 已删去 `gender / age / address`；
  - 已删去同一医生同一问题多次回答辅助布尔字段；
  - 已明确顺序口径统一使用 `time_seq`；
  - 已按 `question_id, time_seq, answer_create_time, seq` 排好序。


## 5. 关键字段说明

### 5.1 标识字段

- `question_id`
  问题唯一标识。
- `user_id`
  医生平台账号标识。
- 当前没有平台真实 `answer_id`。

### 5.2 回答唯一性口径

由于原始数据和代码库中均未发现真实 `answer_id`、回答链接或后台回答 ID，因此已建立代理事件键：

- 文件：
  `data/analysis/base_answer_library_core_physician_round2.parquet`
- 字段：
  `answer_event_proxy_key`
- 构造方法：
  - `question_id`
  - `user_id`
  - `answer_create_time_iso`
  - `answer_text_hash_raw`

注意：

- 这个代理键只用于复现与定位；
- 它不是平台真实回答 ID。

### 5.3 顺序字段

#### `seq`

- 微医原始抓取/展示顺序。
- 不能直接当作真实时间顺序使用。

#### `time_seq`

- 同一 `question_id` 内，按 `answer_create_time` 从早到晚重新排序后的位次。
- 后续所有“题内先后顺序”“前序回答集合”“模板构造”都应以此为准。

#### `observed_seq_by_time`

- 历史审计别名字段；
- 当前实现里与 `time_seq` 完全相同；
- 后续不要再把它当作主字段使用。

### 5.4 前序相关字段

- `prior_answer_count`
  基于原始 `seq` 推得的前序回答数。
- `prior_answer_count_by_time`
  基于 `time_seq` 推得的前序回答数。
- `has_prior_answers`
  是否至少有一个前序回答。

注意：

- 在已经做过医院等级删样本的 `major5` 版本里，如果要计算“当前样本内的严格前序人数”，不要直接套用上游字段，必须重新计算。


## 6. 目前已经完成的工作

以下内容已经完成：

### 6.1 数据审计与目录整理

- 建立标准科研工程目录；
- 只保留 `answer/question` 两表作为主流程；
- 生成文件级审计与字段级审计；
- 完成字段说明补全与字段分布画像；
- 完成基础 EDA。

### 6.2 问答拼接与时间清洗

- 已完成 `question + answer` 拼接；
- 已生成：
  `data/staging/question_answer_joined.parquet`
- 已完成时间顺序检查；
- 已构造：
  - `time_seq`
  - `prior_answer_count_by_time`
  - `answer_delay_hours`
  - `is_seq_time_aligned`
- 已删除 `193` 条时间异常记录。

### 6.3 职称筛选

- 已确认 `tech_title` 基本是一行一标签；
- 已筛到五类核心临床医师职称。

### 6.4 医院等级处理

- 已用 `wy_hospital.parquet` 完成首轮连接；
- 已做第一轮人工补填；
- 已做第二轮 Excel 回填；
- 已形成 `round2` 版本；
- 已进一步筛出五类主要医院等级形成 `major5`。

### 6.5 基础回答库

- 已生成：
  `data/analysis/base_answer_library_core_physician_round2.parquet`
- 检测结果：
  - 未发现“同一记录重复抓取”的精确重复；
  - 已识别模板文本复用。

### 6.6 major5 文本准备

- 已生成：
  `data/analysis/major5_text_cleaning_prep.parquet`
- 已确认：
  - `question_id × user_id` 在当前样本内全部只出现 1 次；
  - 即当前主样本不存在“同一医生在同一问题下多次回答”。

### 6.7 顺序清洗

- 已生成：
  `data/analysis/major5_text_cleaning_prep_seq_clean.parquet`
- 已删除：
  - `gender`
  - `age`
  - `address`
  - `is_same_doctor_multi_answer_in_question`
  - `is_doctor_first_answer_in_question`
  - `is_doctor_followup_answer_in_question`
- 已明确：
  题内顺序统一看 `time_seq`。

### 6.8 严格前序统计

已对当前 `major5_text_cleaning_prep_seq_clean.parquet` 重新计算严格前序口径：

- 总回答数：`2367171`
- 总问题数：`270317`
- 同时间戳回答数：`14167`
- 存在同时间戳回答的问题数：`6094`
- 严格前序至少 `1` 人的焦点回答数：`2096182`
- 严格前序至少 `2` 人的焦点回答数：`1862912`

对应文件：

- `outputs/audit/16_major5_strict_prior_audit_summary.json`
- `outputs/tables/16_strict_prior_count_distribution_major5.csv`
- `outputs/tables/16_same_timestamp_groups_major5.csv`


## 7. 关键脚本与用途

### 上游清洗与构表

- `scripts/02_standardize.py`
  问答拼接。
- `scripts/03_clean_answers.py`
  构造 `time_seq`、时间检查、前序字段。
- `scripts/03d_drop_invalid_time_rows.py`
  删除时间异常样本。
- `scripts/03e_filter_core_physician_titles.py`
  保留五类核心医师。

### 医院等级相关

- `scripts/04b_join_hospital_level.py`
  首轮医院等级连接。
- `scripts/04c_apply_manual_hospitallevel_fill.py`
  第一轮人工补填。
- `scripts/04d_extract_nonstandard_hospital_levels.py`
  导出非标准医院等级供人工搜索。
- `scripts/04e_apply_second_round_hospitallevel_fill.py`
  第二轮 Excel/CSV 回填。
- `scripts/04f_filter_major_hospital_levels.py`
  保留五类主要医院等级，形成 `major5`。

### 回答库与文本准备

- `scripts/04g_build_base_answer_library.py`
  构建基础回答库、代理事件键、重复抓取核验、模板文本识别。
- `scripts/05_prepare_major5_text_base.py`
  生成 `major5_text_cleaning_prep.parquet`。
- `scripts/05b_clean_major5_text_base_for_seq.py`
  删除不需要字段，统一按 `time_seq` 排序。
- `scripts/05c_audit_major5_strict_prior.py`
  统计严格前序人数与同时间戳情况。


## 8. 下一个 AI 现在最应该用哪个文件

### 如果目标是“回答复现 / 去重 / 模板识别”

使用：

- `data/analysis/base_answer_library_core_physician_round2.parquet`

### 如果目标是“主回归/主分析样本”

使用：

- `data/analysis/focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_major5.parquet`

### 如果目标是“继续做文本清洗 / 前序回答模板 / 文本特征计算”

优先使用：

- `data/analysis/major5_text_cleaning_prep_seq_clean.parquet`

这是当前最推荐继续干活的版本。


## 9. 当前版本之间的关系

建议下一个 AI 记住下面这条链：

1. `focal_answer_panel_base_time_valid.parquet`
2. `focal_answer_panel_base_time_valid_core_physician_titles.parquet`
3. `focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel.parquet`
4. `focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_filled.parquet`
5. `focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_filled_round2.parquet`
6. `focal_answer_panel_base_time_valid_core_physician_titles_with_hospitallevel_major5.parquet`
7. `major5_text_cleaning_prep.parquet`
8. `major5_text_cleaning_prep_seq_clean.parquet`

其中：

- 第 5 步是分析母表；
- 第 6 步是主回归样本；
- 第 8 步是当前文本工作主入口。


## 10. 已知重要事实

### 10.1 当前主样本中，同一医生同一问题没有多次回答

在 `major5_text_cleaning_prep.parquet` / `major5_text_cleaning_prep_seq_clean.parquet` 中：

- `question_id × user_id` 全部唯一；
- 不存在同一医生在同一问题下多次回答。

因此当前文本模板工作可以直接把“一条回答”作为最小分析单元。

### 10.2 模板回答复用现象明显

基础回答库中已检测到模板文本复用：

- 模板复用回答行数：`42474`
- 模板复用文本组数：`10462`

因此后续文本清洗时，建议保留模板标记，并考虑向量复用策略。

### 10.3 严格前序必须按当前样本重新算

因为 `major5` 已删过医院等级样本，所以上游继承来的 `time_seq` 虽然仍然代表原始时间位次，但如果研究问题要用“当前样本内严格前序人数”，必须重新按当前样本计算，不能机械使用旧字段。


## 11. 下一个 AI 最推荐的直接起点

如果下一位 AI 要继续推进文本工作，建议直接从：

- `data/analysis/major5_text_cleaning_prep_seq_clean.parquet`

开始，并优先做以下事项：

1. 新增 `strict_prior_count_current` 到新的 parquet 版本；
2. 新增 `same_timestamp_group_size` 到新的 parquet 版本；
3. 开始真正的文本清洗字段构造，例如：
   - `answer_clean`
   - `question_desc_clean`
   - 去空白版
   - 去换行版
   - 模板引导类文本标记
4. 在严格前序口径下构造前序回答集合；
5. 为后续主题、相似度、知识分化度、多样性计算做准备。


## 12. 最后提醒

下一个 AI 不要做以下错误操作：

- 不要修改 `data/raw/`；
- 不要把 `seq` 当作真实时间顺序；
- 不要把 `observed_seq_by_time` 当成独立于 `time_seq` 的新信息；
- 不要把 `major5` 样本和基础回答库混用；
- 不要把代理事件键误当作平台真实 `answer_id`。

当前最安全的理解是：

- 基础回答库负责“回答事件保存与定位”；
- 分析母表负责“可再派生”；
- `major5` 负责“主分析样本”；
- `major5_text_cleaning_prep_seq_clean.parquet` 负责“继续往下做文本工作”。
