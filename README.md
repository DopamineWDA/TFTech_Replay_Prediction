# IFTech 算法笔试题提交说明

本仓库用于完成 IFTech 算法笔试题“预测用户重复播放的概率”。目标是基于用户首次播放播客单集时的上下文、用户静态信息和单集静态信息，预测该用户在未来一个月内是否会对该单集发生重复播放行为，并输出测试集概率结果。

## 题目要求交付文件

以下内容均已包含在本仓库中，可直接按路径查看：

- 提交结果文件：`result.csv`
- 特征工程代码：`build_baseline_features.py`
- 模型训练代码：`train_lightgbm_gpu.py`
- 运行说明：`README.md`
- 分析报告：`analysis.md`

题目原始要求文件路径：

- `doc/task.md`

## 补充说明文档路径

`doc/` 目录中提供了本次提交的补充说明材料：

- 题目原文：`doc/task.md`
- EDA 摘要：`doc/eda_summary.md`
- 特征清单：`doc/feature_catalog.md`
- 特征工程实现说明：`doc/feature_result.md`
- GPU 环境记录：`doc/LGBM_GPU_ENV.md`

## 项目结构

- `data/`：原始数据文件
- `build_baseline_features.py`：特征工程脚本
- `eda_basic.py`：EDA 脚本
- `train_lightgbm_gpu.py`：模型训练脚本
- `feature/`：生成后的训练特征与测试特征
- `lgbm_runs/run_seed42_gpu_output/`：模型输出目录
- `analysis.md`：主分析报告
- `doc/`：补充说明文档

## 环境依赖

本次提交使用 Python 3.9 环境，主要依赖如下：

- `lightgbm`
- `pandas`
- `numpy`
- `scikit-learn`

GPU 训练环境记录见 `doc/LGBM_GPU_ENV.md`。

## 运行步骤

### 1. 执行 EDA

```bash
python eda_basic.py
```

该脚本会读取原始数据，输出数据规模、缺失情况、类别分布、用户与单集统计等基础结论，用于支持后续特征设计。

### 2. 生成特征

```bash
python build_baseline_features.py
```

脚本默认配置如下：

- 输入目录：`data`
- 输出目录：`feature`
- OOF target encoding 折数：`3`

生成文件：

- `feature/train_features_baseline.csv`
- `feature/test_features_baseline.csv`

### 3. 训练模型并生成提交结果

```bash
python train_lightgbm_gpu.py
```

本次提交所使用的默认配置如下：

- `device = gpu`
- `gpu_id = 0`
- `n_splits = 3`
- `num_boost_round = 1500`
- `early_stopping_rounds = 100`
- `learning_rate = 0.03`
- `num_leaves = 255`
- `feature_fraction = 0.8`
- `bagging_fraction = 0.8`
- `bagging_freq = 1`
- `min_data_in_leaf = 100`
- `output_dir = lgbm_runs/run_seed42_gpu_output`

CPU 运行命令如下：

```bash
python train_lightgbm_gpu.py --device cpu
```

## 输出结果

训练完成后，输出目录 `lgbm_runs/run_seed42_gpu_output/` 中包含：

- `result.csv`：测试集提交文件
- `test_predictions.csv`：测试集预测结果备份
- `oof_predictions.csv`：训练集 OOF 预测
- `cv_metrics.json`：交叉验证指标
- `feature_importance_mean.csv`：平均特征重要性
- `feature_importance_folds.csv`：分折特征重要性
- `fold_0.txt`
- `fold_1.txt`
- `fold_2.txt`

## 本次提交结果

本次提交使用 3 折 GPU LightGBM，交叉验证结果如下：

- `CV AUC = 0.819658`
- `CV Logloss = 0.517444`

最终提交文件路径：

- `lgbm_runs/run_seed42_gpu_output/result.csv`

## 提交说明

本仓库主报告为 `analysis.md`，`doc/` 目录用于补充说明 EDA、特征设计、特征工程实现细节和 GPU 环境信息。
