# IFTech Baseline 特征清单

本清单用于说明本次提交版本中实际纳入模型训练的特征结构。

## 1. 主键与目标字段

- 训练集主键：`row_id`
- 测试集主键：`id`
- 目标字段：`label`

说明：

- `row_id` 由特征工程脚本生成，用于训练样本顺序对齐与 OOF 结果回溯。
- `id` 保留测试集原始主键，用于生成最终 `result.csv`。

## 2. 原始行为上下文字段

来自 `train.csv` / `test.csv`：

- `uid`
- `episode_id`
- `tab_name`
- `scene_name`
- `entrance_type`

处理方式：

- 类别缺失值统一填充为 `MISSING`
- `tab_name`、`scene_name`、`entrance_type` 作为模型类别特征使用
- `uid`、`episode_id` 主要通过统计特征表达

## 3. 用户侧特征

来源：`user_feature.csv`

### 3.1 类别特征

- `address`
- `sex`
- `rg_source`

### 3.2 数值与分桶特征

- `age`
- `age_bucket`
- `rg_year`
- `rg_month`
- `exp_year`
- `exp_month`
- `membership_days`
- `is_age_zero`
- `is_age_outlier`
- `is_membership_days_negative`

处理方式：

- `age=0` 单独标记为未知年龄
- 异常年龄和异常会员时长保留并显式标记
- 时间字段拆解为年月信息

## 4. 单集侧特征

来源：`episode_feature.csv` + `episode_additional.csv`

### 4.1 类别特征

- `language`
- `duration_bucket`

### 4.2 数值特征

- `duration_ms`
- `duration_sec`
- `duration_log1p`
- `category_count`
- `host_count`
- `producer_count`
- `writer_count`
- `title_token_count`
- `title_char_len`
- `uuid_episode_count`

### 4.3 标识与浅层结构字段

- `uuid`

处理方式：

- `duration` 同时保留原值、秒级值、对数值和区间分桶
- 多值字段使用数量统计而非完整展开
- `title` 使用浅层统计特征
- `uuid_episode_count` 反映同一 `uuid` 关联单集数量

## 5. 无监督频次统计特征

### 5.1 单字段频次

- `uid_count`
- `episode_count`
- `tab_name_count`
- `scene_name_count`
- `entrance_type_count`

### 5.2 交叉频次

- `uid_tab_count`
- `uid_entrance_count`
- `episode_tab_count`
- `episode_entrance_count`

说明：

- 这类特征不依赖标签，可直接用训练集统计得到。
- 它们主要表达用户活跃度、单集热度和行为组合熟悉度。

## 6. OOF target encoding 特征

### 6.1 单字段 OOF target encoding

- `te_uid`
- `te_episode_id`
- `te_tab_name`
- `te_scene_name`
- `te_entrance_type`

### 6.2 交叉字段 OOF target encoding

- `te_uid_tab`
- `te_uid_entrance`
- `te_episode_tab`
- `te_episode_entrance`

### 6.3 辅助字段

- `te_uid_train_count`
- `te_episode_id_train_count`
- `te_tab_name_train_count`
- `te_scene_name_train_count`
- `te_entrance_type_train_count`
- `te_uid_tab_train_count`
- `te_uid_entrance_train_count`
- `te_episode_tab_train_count`
- `te_episode_entrance_train_count`
- `global_label_mean`

说明：

- 训练集采用 OOF 方式生成 target encoding
- 测试集采用全训练集统计结果生成
- 稀疏分组回退到全局均值

## 7. 本次提交未纳入的特征

以下内容未纳入本次正式提交版本：

- `title` 语义 NLP 特征
- `host` / `producer` / `writer` 的完整多热展开
- 更高阶三元或四元组合特征
- embedding 或深度模型特征

原因：

- 当前版本以训练稳定、可解释和工程可复现为主
- 现有 EDA 已证明上下文类别、频次统计和 OOF target encoding 构成主要信息来源

## 8. 特征工程输出文件

本次提交版本的特征工程脚本输出：

- `IFTech/feature/train_features_baseline.csv`
- `IFTech/feature/test_features_baseline.csv`

字段设计原则：

- 训练集与测试集尽量保持一致的特征列
- 训练集额外保留 `row_id` 与 `label`
- 测试集额外保留 `id`

## 9. 泄漏控制原则

允许直接使用的统计：

- 频次特征
- 分桶特征
- 缺失标记
- 原始静态 join 特征

必须 OOF 生成的统计：

- 任意基于 `label` 的均值统计
- 包括单字段和交叉字段 target encoding

## 10. 本次模型适配关系

这套特征当前直接用于：

- `LightGBM`

同时保留了向其他树模型扩展的空间，但本次正式提交版本只使用了 LightGBM 结果。
