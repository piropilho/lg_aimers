"""XGBoost / LightGBM / CatBoost 하이퍼파라미터 튜닝.

전략:
  - 서치 단계: train(2019-2023)에서 25% 샘플로 빠르게 Optuna 탐색 (early stopping은 항상 전체 validation=2024)
  - 최종 단계: 찾은 best params로 전체 train에 재학습 후 validation AUC/LogLoss 기록
  - 결과는 tuning_results.json에 저장
"""

import json
import time

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from features import DROP_COLS, ENG_CAT_COLS, TARGET_COL, build_matrix, engineer_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = "../data"
RANDOM_STATE = 42
SEARCH_SAMPLE_SIZE = 300_000
N_TRIALS = {"xgboost": 40, "lightgbm": 40, "catboost": 20}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    log("Loading train.csv ...")
    train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    train_eng = engineer_features(train_raw)
    feature_cols = [c for c in train_eng.columns if c not in DROP_COLS]
    y = train_eng[TARGET_COL]
    train_mask = train_eng["season"] < 2024
    log(f"features={len(feature_cols)}  train={train_mask.sum()}  valid={(~train_mask).sum()}")
    return train_eng, feature_cols, y, train_mask


def main():
    train_eng, feature_cols, y, train_mask = load_data()

    X_tree = build_matrix(train_eng, feature_cols, ENG_CAT_COLS, mode="tree")
    X_cb = build_matrix(train_eng, feature_cols, ENG_CAT_COLS, mode="cb")

    X_train_tree, X_valid_tree = X_tree[train_mask], X_tree[~train_mask]
    X_train_cb, X_valid_cb = X_cb[train_mask], X_cb[~train_mask]
    y_train, y_valid = y[train_mask], y[~train_mask]

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(X_train_tree.index, size=min(SEARCH_SAMPLE_SIZE, len(X_train_tree)), replace=False)
    X_search_tree = X_train_tree.loc[sample_idx]
    X_search_cb = X_train_cb.loc[sample_idx]
    y_search = y_train.loc[sample_idx]
    log(f"search sample size={len(sample_idx)}")

    results = {}

    # ---------------- XGBoost ----------------
    def objective_xgb(trial):
        params = dict(
            n_estimators=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 20, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            tree_method="hist", enable_categorical=True,
            eval_metric="logloss", early_stopping_rounds=30,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        model = xgb.XGBClassifier(**params)
        model.fit(X_search_tree, y_search, eval_set=[(X_valid_tree, y_valid)], verbose=False)
        pred = model.predict_proba(X_valid_tree)[:, 1]
        return roc_auc_score(y_valid, pred)

    log("Tuning XGBoost ...")
    t0 = time.time()
    study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(objective_xgb, n_trials=N_TRIALS["xgboost"], show_progress_bar=False)
    log(f"XGBoost search done in {time.time()-t0:.1f}s  best_auc(sample)={study_xgb.best_value:.5f}")

    final_xgb = xgb.XGBClassifier(
        **study_xgb.best_params, n_estimators=2000,
        tree_method="hist", enable_categorical=True,
        eval_metric="logloss", early_stopping_rounds=50,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    t0 = time.time()
    final_xgb.fit(X_train_tree, y_train, eval_set=[(X_valid_tree, y_valid)], verbose=False)
    fit_time = time.time() - t0
    pred = final_xgb.predict_proba(X_valid_tree)[:, 1]
    results["xgboost"] = {
        "best_params": study_xgb.best_params,
        "final_auc": roc_auc_score(y_valid, pred),
        "final_logloss": log_loss(y_valid, pred),
        "best_iteration": int(final_xgb.best_iteration),
        "full_fit_time_s": fit_time,
    }
    log(f"XGBoost FINAL — AUC={results['xgboost']['final_auc']:.5f}  LogLoss={results['xgboost']['final_logloss']:.5f}")

    # ---------------- LightGBM ----------------
    def objective_lgb(trial):
        params = dict(
            n_estimators=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_search_tree, y_search,
            eval_set=[(X_valid_tree, y_valid)], eval_metric="logloss",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        pred = model.predict_proba(X_valid_tree)[:, 1]
        return roc_auc_score(y_valid, pred)

    log("Tuning LightGBM ...")
    t0 = time.time()
    study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgb.optimize(objective_lgb, n_trials=N_TRIALS["lightgbm"], show_progress_bar=False)
    log(f"LightGBM search done in {time.time()-t0:.1f}s  best_auc(sample)={study_lgb.best_value:.5f}")

    final_lgb = lgb.LGBMClassifier(
        **study_lgb.best_params, n_estimators=2000,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    t0 = time.time()
    final_lgb.fit(
        X_train_tree, y_train,
        eval_set=[(X_valid_tree, y_valid)], eval_metric="logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    fit_time = time.time() - t0
    pred = final_lgb.predict_proba(X_valid_tree)[:, 1]
    results["lightgbm"] = {
        "best_params": study_lgb.best_params,
        "final_auc": roc_auc_score(y_valid, pred),
        "final_logloss": log_loss(y_valid, pred),
        "best_iteration": int(final_lgb.best_iteration_),
        "full_fit_time_s": fit_time,
    }
    log(f"LightGBM FINAL — AUC={results['lightgbm']['final_auc']:.5f}  LogLoss={results['lightgbm']['final_logloss']:.5f}")

    # ---------------- CatBoost ----------------
    def objective_cb(trial):
        params = dict(
            iterations=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 30, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
            cat_features=ENG_CAT_COLS, eval_metric="Logloss",
            random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=30,
        )
        model = CatBoostClassifier(**params)
        model.fit(X_search_cb, y_search, eval_set=(X_valid_cb, y_valid))
        pred = model.predict_proba(X_valid_cb)[:, 1]
        return roc_auc_score(y_valid, pred)

    log("Tuning CatBoost ...")
    t0 = time.time()
    study_cb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_cb.optimize(objective_cb, n_trials=N_TRIALS["catboost"], show_progress_bar=False)
    log(f"CatBoost search done in {time.time()-t0:.1f}s  best_auc(sample)={study_cb.best_value:.5f}")

    final_cb = CatBoostClassifier(
        **study_cb.best_params, iterations=3000,
        cat_features=ENG_CAT_COLS, eval_metric="Logloss",
        random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=50,
    )
    t0 = time.time()
    final_cb.fit(X_train_cb, y_train, eval_set=(X_valid_cb, y_valid))
    fit_time = time.time() - t0
    pred = final_cb.predict_proba(X_valid_cb)[:, 1]
    results["catboost"] = {
        "best_params": study_cb.best_params,
        "final_auc": roc_auc_score(y_valid, pred),
        "final_logloss": log_loss(y_valid, pred),
        "best_iteration": int(final_cb.get_best_iteration()),
        "full_fit_time_s": fit_time,
    }
    log(f"CatBoost FINAL — AUC={results['catboost']['final_auc']:.5f}  LogLoss={results['catboost']['final_logloss']:.5f}")

    with open("tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("Saved tuning_results.json")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
