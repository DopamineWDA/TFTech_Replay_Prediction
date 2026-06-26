# Baseline 特征工程实现说明

本文档说明 `IFTech/build_baseline_features.py` 在本次提交版本中完成了哪些工作，以及输入、输出和关键实现逻辑。

## 1. 目标

本次特征工程的目标是：

- 将原始行为样本、用户静态信息和单集静态信息整合成统一训练样本
- 构造能够直接供 LightGBM 训练的结构化特征
- 在引入监督统计特征时严格控制标签泄漏

## 2. 输入文件

脚本读取以下 5 个文件：

- `IFTech/data/train.csv`
- `IFTech/data/test.csv`
- `IFTech/data/user_feature.csv`
- `IFTech/data/episode_feature.csv`
- `IFTech/data/episode_additional.csv`

其中：

- `train.csv` 提供训练样本与 `label`
- `test.csv` 提供测试样本与 `id`
- `user_feature.csv` 提供用户静态画像
- `episode_feature.csv` 提供单集静态结构信息
- `episode_additional.csv` 提供 `title` 与 `uuid`

## 3. 输出文件

脚本最终输出：

- `IFTech/feature/train_features_baseline.csv`
- `IFTech/feature/test_features_baseline.csv`

两张表的关系为：

- 特征列保持一致
- 训练集额外保留 `row_id` 与 `label`
- 测试集额外保留 `id`

## 4. 整体流程

本次特征工程可以拆成 6 个步骤：

1. 收集训练集和测试集中实际涉及的 `episode_id`
2. 读取并构造用户侧特征映射
3. 读取并构造单集侧特征映射
4. 在训练集上预先统计频次特征与 target encoding 所需统计量
5. 逐行拼接训练特征和测试特征
6. 写出最终 CSV 文件

## 5. 第一步：筛选需要的 `episode_id`

脚本先遍历：

- `train.csv`
- `test.csv`

收集所有实际出现过的 `episode_id`。

目的如下：

- 仅加载后续真正会用到的单集侧记录
- 降低 `episode_feature.csv` 与 `episode_additional.csv` 的读取和内存压力

这一步是面向大规模数据的工程优化。

## 6. 第二步：构造用户侧特征

脚本将 `user_feature.csv` 转为：

- `uid -> 用户特征字典`

每个用户生成的字段包括：

- `address`
- `sex`
- `rg_source`
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

关键处理逻辑如下：

- 类别缺失值统一填充为 `MISSING`
- `age=0` 保留原值，同时作为未知年龄标记
- 极端年龄值保留原值，并通过 `is_age_outlier` 标记
- `membership_days = exp_date - rg_date`
- 对 `membership_days < 0` 的记录使用异常标记保留信息

## 7. 第三步：构造单集侧特征

脚本将 `episode_feature.csv` 与 `episode_additional.csv` 合并为：

- `episode_id -> 单集特征字典`

### 7.1 来自 `episode_feature.csv` 的特征

- `language`
- `duration_ms`
- `duration_sec`
- `duration_log1p`
- `duration_bucket`
- `category_count`
- `host_count`
- `producer_count`
- `writer_count`

关键处理逻辑：

- 将 `duration` 同时保留为原值、秒级值、对数值和区间分桶
- `category_ids`、`host`、`producer`、`writer` 均使用数量统计

### 7.2 来自 `episode_additional.csv` 的特征

- `title_token_count`
- `title_char_len`
- `uuid`
- `uuid_episode_count`

关键处理逻辑：

- `title` 不做语义建模，只做长度和 token 数统计
- 先统计每个 `uuid` 在数据中出现的次数，再映射为 `uuid_episode_count`

## 8. 第四步：生成基础特征行

对主表中每一条记录，脚本会：

1. 读取原始行为字段：
   - `uid`
   - `episode_id`
   - `tab_name`
   - `scene_name`
   - `entrance_type`
2. 用 `uid` 关联用户侧特征
3. 用 `episode_id` 关联单集侧特征
4. 拼出一条完整的基础特征行

在这一阶段：

- 所有空类别字段统一转为 `MISSING`
- 尚未引入任何基于标签的监督统计

## 9. 第五步：构造无监督频次特征

脚本在训练集上统计以下频次特征。

### 9.1 单字段频次

- `uid_count`
- `episode_count`
- `tab_name_count`
- `scene_name_count`
- `entrance_type_count`

### 9.2 交叉频次

- `uid_tab_count`
- `uid_entrance_count`
- `episode_tab_count`
- `episode_entrance_count`

这些特征不依赖标签，因此不会引入泄漏，但可以有效表达：

- 用户活跃度
- 单集热度
- 特定行为上下文下的出现频次

## 10. 第六步：构造 OOF target encoding

脚本生成如下监督统计特征。

### 10.1 单字段 OOF target encoding

- `te_uid`
- `te_episode_id`
- `te_tab_name`
- `te_scene_name`
- `te_entrance_type`

### 10.2 交叉字段 OOF target encoding

- `te_uid_tab`
- `te_uid_entrance`
- `te_episode_tab`
- `te_episode_entrance`

### 10.3 辅助字段

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

辅助计数字段表示当前编码所依赖的样本量，用于体现编码可靠性。

## 11. 泄漏控制机制

如果直接在全训练集上计算某个分组的标签均值，再回填给训练集本身，就会发生 target leakage。

例如：

- 某个 `episode_id` 在训练集中只出现 1 次
- 标签为 `1`
- 直接全量编码会得到 `te_episode_id = 1.0`

模型实际上就等于直接读取了答案。

为避免这个问题，脚本使用 OOF 思路：

1. 使用稳定哈希方式为训练样本分配 fold
2. 统计全体训练样本的标签和与样本数
3. 统计当前 fold 的标签和与样本数
4. 用“全体统计 - 当前 fold 统计”生成当前样本可用编码

测试集没有标签，因此直接使用全训练集统计结果。

## 12. fold 划分方式

脚本没有额外依赖 `KFold` 生成器，而是使用以下字段构造稳定哈希键：

- `uid`
- `episode_id`
- `row_idx`

处理方式为：

- 字段拼接
- `md5`
- 对 `n_splits` 取模

这种方式的特点是：

- 可复现
- 实现简单
- 适合在大文件流式处理场景下使用

## 13. 输出列结构

最终输出列可分为四层。

### 13.1 标识与目标列

训练集：

- `row_id`
- `label`

测试集：

- `id`

### 13.2 原始与浅层类别列

- `uid`
- `episode_id`
- `tab_name`
- `scene_name`
- `entrance_type`
- `address`
- `sex`
- `rg_source`
- `age_bucket`
- `language`
- `duration_bucket`
- `uuid`

### 13.3 数值特征列

- `age`
- `rg_year`
- `rg_month`
- `exp_year`
- `exp_month`
- `membership_days`
- `is_age_zero`
- `is_age_outlier`
- `is_membership_days_negative`
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

### 13.4 统计特征列

- 频次特征
- OOF target encoding
- target encoding 对应样本量
- `global_label_mean`

## 14. 本次实现的特点

本次特征工程实现具有以下特点：

- 面向数百万级样本可运行
- 特征逻辑与 EDA 结论保持一致
- 对异常值和缺失值采用保留并标记的处理方式
- 对高价值监督统计采用 OOF 控制泄漏
- 输出结果可直接衔接 LightGBM 训练

## 15. 总结

这份特征工程脚本本质上完成的是：

将原始行为样本转化为“行为上下文 + 用户画像 + 单集画像 + 频次统计 + 不泄漏监督统计”的结构化样本表，并为后续模型训练提供稳定输入。
