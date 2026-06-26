# LightGBM GPU 环境记录

## 1. 环境位置

- 虚拟环境目录：`IFTech/.venv-lgbm-gpu`

## 2. 主要依赖

- `lightgbm==4.6.0`
- `pandas`
- `scikit-learn`

## 3. 环境激活

```bash
source IFTech/.venv-lgbm-gpu/bin/activate
```

## 4. 环境自检

### 4.1 检查 LightGBM 导入

```bash
IFTech/.venv-lgbm-gpu/bin/python - <<'PY'
import lightgbm as lgb
print(lgb.__version__)
PY
```

### 4.2 检查 GPU 训练能力

```bash
IFTech/.venv-lgbm-gpu/bin/python - <<'PY'
import numpy as np
import lightgbm as lgb
X = np.random.rand(2000, 10).astype("float32")
y = np.random.randint(0, 2, 2000)
train = lgb.Dataset(X, label=y)
booster = lgb.train(
    {
        "objective": "binary",
        "metric": "auc",
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "verbosity": -1,
    },
    train,
    num_boost_round=5,
)
print("GPU_TRAIN_OK")
PY
```

## 5. 本次提交使用的训练命令

```bash
IFTech/.venv-lgbm-gpu/bin/python IFTech/train_lightgbm_gpu.py \
  --device gpu \
  --gpu-id 0 \
  --output-dir IFTech/lgbm_runs/run_seed42_gpu_output \
  --n-splits 3 \
  --num-boost-round 1500 \
  --early-stopping-rounds 100
```

## 6. 说明

本次提交版本使用单卡 GPU 训练，设备编号为 `0`。训练脚本中的 `gpu_id` 参数用于显式指定训练设备，以保证结果目录和训练配置与提交版本保持一致。
