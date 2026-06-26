import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


CATEGORICAL_COLS = [
    "tab_name",
    "scene_name",
    "entrance_type",
    "address",
    "sex",
    "rg_source",
    "age_bucket",
    "language",
    "duration_bucket",
]

HIGH_CARD_RAW_ID_COLS = [
    "uid",
    "episode_id",
    "uuid",
]

IGNORE_TRAIN_COLS = {"row_id", "label"}
IGNORE_TEST_COLS = {"id"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM on IFTech baseline features.")
    parser.add_argument(
        "--train-path",
        default="IFTech/feature/train_features_baseline.csv",
        help="Path to train baseline features csv.",
    )
    parser.add_argument(
        "--test-path",
        default="IFTech/feature/test_features_baseline.csv",
        help="Path to test baseline features csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="IFTech/lgbm_runs/run_seed42_gpu_output",
        help="Directory for models, predictions, and reports.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--num-boost-round", type=int, default=1500)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=255)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--min-data-in-leaf", type=int, default=100)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for smoke tests.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device id to use.")
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="LightGBM device.",
    )
    parser.add_argument(
        "--keep-raw-id-features",
        action="store_true",
        help="Keep raw uid / episode_id / uuid as model features. Off by default because GPU LightGBM is sensitive to very high-cardinality categoricals.",
    )
    return parser.parse_args()


def load_csvs(train_path, test_path, max_rows=0):
    nrows = max_rows if max_rows > 0 else None
    train_df = pd.read_csv(train_path, nrows=nrows, low_memory=False)
    test_df = pd.read_csv(test_path, nrows=nrows, low_memory=False)
    return train_df, test_df


def align_categoricals(train_df, test_df, categorical_cols):
    for col in categorical_cols:
        train_df[col] = train_df[col].astype("string").fillna("MISSING")
        test_df[col] = test_df[col].astype("string").fillna("MISSING")
        categories = pd.Index(train_df[col].unique()).union(pd.Index(test_df[col].unique()))
        train_df[col] = pd.Categorical(train_df[col], categories=categories)
        test_df[col] = pd.Categorical(test_df[col], categories=categories)
    return train_df, test_df


def optimize_numeric_dtypes(df, feature_cols, categorical_cols):
    cat_set = set(categorical_cols)
    for col in feature_cols:
        if col in cat_set:
            continue
        if col.startswith("te_") and not col.endswith("_train_count"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        elif col in {"duration_sec", "duration_log1p", "global_label_mean"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        else:
            series = pd.to_numeric(df[col], errors="coerce")
            if pd.api.types.is_float_dtype(series):
                df[col] = series.astype("float32")
            else:
                df[col] = series.astype("int32")
    return df


def cast_categoricals_after_numeric(train_df, test_df, categorical_cols):
    for col in categorical_cols:
        train_df[col] = train_df[col].astype("string").fillna("MISSING")
        test_df[col] = test_df[col].astype("string").fillna("MISSING")
        categories = pd.Index(train_df[col].unique()).union(pd.Index(test_df[col].unique()))
        train_df[col] = pd.Categorical(train_df[col], categories=categories)
        test_df[col] = pd.Categorical(test_df[col], categories=categories)
    return train_df, test_df


def prepare_data(train_df, test_df, keep_raw_id_features=False):
    feature_cols = [c for c in train_df.columns if c not in IGNORE_TRAIN_COLS]
    if not keep_raw_id_features:
        feature_cols = [c for c in feature_cols if c not in HIGH_CARD_RAW_ID_COLS]
    train_df = optimize_numeric_dtypes(train_df, feature_cols, CATEGORICAL_COLS)
    test_df = optimize_numeric_dtypes(test_df, feature_cols, CATEGORICAL_COLS)
    train_df, test_df = cast_categoricals_after_numeric(train_df, test_df, CATEGORICAL_COLS)
    return feature_cols, train_df, test_df


def build_params(args):
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "min_data_in_leaf": args.min_data_in_leaf,
        "max_bin": args.max_bin,
        "verbosity": -1,
        "seed": args.seed,
        "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
        "data_random_seed": args.seed,
        "deterministic": True,
        "force_col_wise": True,
    }
    if args.device == "gpu":
        params.update(
            {
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": args.gpu_id,
            }
        )
    else:
        params["device"] = "cpu"
    return params


def train_cv(train_df, test_df, feature_cols, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = train_df[feature_cols]
    y = train_df["label"].astype("uint8").to_numpy()
    X_test = test_df[feature_cols]
    test_ids = test_df["id"].copy()

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    params = build_params(args)

    oof_pred = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)
    fold_metrics = []
    fold_importances = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"=== Fold {fold} / device={args.device} ===")
        X_train = X.iloc[trn_idx]
        y_train = y[trn_idx]
        X_valid = X.iloc[val_idx]
        y_valid = y[val_idx]

        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=CATEGORICAL_COLS,
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            categorical_feature=CATEGORICAL_COLS,
            free_raw_data=False,
        )

        booster = lgb.train(
            params=params,
            train_set=train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(100),
            ],
        )

        valid_pred = booster.predict(X_valid, num_iteration=booster.best_iteration)
        fold_test_pred = booster.predict(X_test, num_iteration=booster.best_iteration)
        oof_pred[val_idx] = valid_pred
        test_pred += fold_test_pred / args.n_splits

        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_logloss = log_loss(y_valid, valid_pred, labels=[0, 1])
        fold_metrics.append(
            {
                "fold": fold,
                "best_iteration": int(booster.best_iteration),
                "auc": float(fold_auc),
                "logloss": float(fold_logloss),
            }
        )

        fold_importances.append(
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "gain": booster.feature_importance(importance_type="gain"),
                    "split": booster.feature_importance(importance_type="split"),
                    "fold": fold,
                }
            )
        )

        booster.save_model(str(output_dir / f"fold_{fold}.txt"))

    full_auc = roc_auc_score(y, oof_pred)
    full_logloss = log_loss(y, oof_pred, labels=[0, 1])

    metrics = {
        "seed": args.seed,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "n_splits": args.n_splits,
        "cv_auc": float(full_auc),
        "cv_logloss": float(full_logloss),
        "fold_metrics": fold_metrics,
    }

    with (output_dir / "cv_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    oof_df = pd.DataFrame(
        {
            "row_id": train_df["row_id"].astype("int64"),
            "label": train_df["label"].astype("int8"),
            "oof_pred": oof_pred,
        }
    )
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    test_pred_df = pd.DataFrame({"id": test_ids.astype("int64"), "label": test_pred})
    test_pred_df.to_csv(output_dir / "result.csv", index=False)
    test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    fi_df = pd.concat(fold_importances, ignore_index=True)
    fi_df.to_csv(output_dir / "feature_importance_folds.csv", index=False)
    fi_summary = (
        fi_df.groupby("feature", as_index=False)[["gain", "split"]]
        .mean()
        .sort_values("gain", ascending=False)
    )
    fi_summary.to_csv(output_dir / "feature_importance_mean.csv", index=False)

    print("CV AUC:", round(full_auc, 6))
    print("CV Logloss:", round(full_logloss, 6))
    print("Saved to:", output_dir)


def main():
    args = parse_args()
    train_df, test_df = load_csvs(args.train_path, args.test_path, max_rows=args.max_rows)
    feature_cols, train_df, test_df = prepare_data(
        train_df,
        test_df,
        keep_raw_id_features=args.keep_raw_id_features,
    )
    train_cv(train_df, test_df, feature_cols, args)


if __name__ == "__main__":
    main()
