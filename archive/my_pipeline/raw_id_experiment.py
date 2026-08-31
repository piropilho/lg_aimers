"""pitcher_id/batter_id를 raw 피처로 그대로 투입하는 효과 검증 (팀원 RF가 쓴 방식 이식).

- XGBoost/LightGBM/CatBoost 공통: pitcher_id/batter_id를 '수치형'으로 그대로 투입 (팀원 RF와 동일 방식)
- CatBoost만 추가로: '범주형'으로 투입해서 CatBoost 자체의 순서형(ordered) 타겟통계 활용 버전도 검증
  (우리가 직접 만든 target encoding/context deviation은 둘 다 실패했는데,
   CatBoost 내장 방식은 leakage 안전장치가 다르게 구현되어 있어 별도로 확인할 가치 있음)
"""

import json
import time

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from features import ENG_CAT_COLS, TARGET_COL, build_matrix, engineer_features

DATA_DIR = "../data"
RANDOM_STATE = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    with open("tuning_results.json", encoding="utf-8") as f:
        tuned = json.load(f)

    log("Loading train.csv ...")
    train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    train_eng = engineer_features(train_raw)
    train_mask = train_eng["season"] < 2024
    y = train_eng[TARGET_COL]

    # DROP_COLS는 기본적으로 pitcher_id/batter_id를 빼는 리스트라, 여기서는 직접 다시 살려서 포함시킨다
    feature_cols_with_ids = [c for c in train_eng.columns if c not in ["row_id", "control_success"]]
    log(f"feature 수: {len(feature_cols_with_ids)} (pitcher_id/batter_id 수치형으로 포함)")

    results = {}

    # ---------------- 실험 A: pitcher_id/batter_id를 수치형으로 (팀원 RF 방식) ----------------
    X_tree = build_matrix(train_eng, feature_cols_with_ids, ENG_CAT_COLS, mode="tree")
    X_cb = build_matrix(train_eng, feature_cols_with_ids, ENG_CAT_COLS, mode="cb")
    X_train_tree, X_valid_tree = X_tree[train_mask], X_tree[~train_mask]
    X_train_cb, X_valid_cb = X_cb[train_mask], X_cb[~train_mask]
    y_train, y_valid = y[train_mask], y[~train_mask]

    log("Training XGBoost (+ raw pitcher_id/batter_id 수치형) ...")
    t0 = time.time()
    model = xgb.XGBClassifier(
        **tuned["xgboost"]["best_params"], n_estimators=2000,
        tree_method="hist", enable_categorical=True,
        eval_metric="logloss", early_stopping_rounds=50,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train_tree, y_train, eval_set=[(X_valid_tree, y_valid)], verbose=False)
    pred = model.predict_proba(X_valid_tree)[:, 1]
    results["xgboost_raw_id_numeric"] = {"auc": roc_auc_score(y_valid, pred), "logloss": log_loss(y_valid, pred), "fit_time_s": time.time() - t0}
    log(f"XGBoost+rawID(수치) — AUC={results['xgboost_raw_id_numeric']['auc']:.5f}  (baseline: {tuned['xgboost']['final_auc']:.5f})")

    log("Training LightGBM (+ raw pitcher_id/batter_id 수치형) ...")
    t0 = time.time()
    model = lgb.LGBMClassifier(
        **tuned["lightgbm"]["best_params"], n_estimators=2000,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    model.fit(
        X_train_tree, y_train,
        eval_set=[(X_valid_tree, y_valid)], eval_metric="logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = model.predict_proba(X_valid_tree)[:, 1]
    results["lightgbm_raw_id_numeric"] = {"auc": roc_auc_score(y_valid, pred), "logloss": log_loss(y_valid, pred), "fit_time_s": time.time() - t0}
    log(f"LightGBM+rawID(수치) — AUC={results['lightgbm_raw_id_numeric']['auc']:.5f}  (baseline: {tuned['lightgbm']['final_auc']:.5f})")

    log("Training CatBoost (+ raw pitcher_id/batter_id 수치형) ...")
    t0 = time.time()
    model = CatBoostClassifier(
        **tuned["catboost"]["best_params"], iterations=3000,
        cat_features=ENG_CAT_COLS, eval_metric="Logloss",
        random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=50,
    )
    model.fit(X_train_cb, y_train, eval_set=(X_valid_cb, y_valid))
    pred = model.predict_proba(X_valid_cb)[:, 1]
    results["catboost_raw_id_numeric"] = {"auc": roc_auc_score(y_valid, pred), "logloss": log_loss(y_valid, pred), "fit_time_s": time.time() - t0}
    log(f"CatBoost+rawID(수치) — AUC={results['catboost_raw_id_numeric']['auc']:.5f}  (baseline: {tuned['catboost']['final_auc']:.5f})")

    # ---------------- 실험 B: CatBoost만 — pitcher_id/batter_id를 범주형으로 (내장 ordered TS 활용) ----------------
    log("Training CatBoost (+ raw pitcher_id/batter_id 범주형, CatBoost 내장 ordered target stats) ...")
    cat_cols_with_ids = ENG_CAT_COLS + ["pitcher_id", "batter_id"]
    X_cb2 = build_matrix(train_eng, feature_cols_with_ids, cat_cols_with_ids, mode="cb")
    X_train_cb2, X_valid_cb2 = X_cb2[train_mask], X_cb2[~train_mask]

    t0 = time.time()
    model = CatBoostClassifier(
        **tuned["catboost"]["best_params"], iterations=3000,
        cat_features=cat_cols_with_ids, eval_metric="Logloss",
        random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=50,
    )
    model.fit(X_train_cb2, y_train, eval_set=(X_valid_cb2, y_valid))
    pred = model.predict_proba(X_valid_cb2)[:, 1]
    results["catboost_raw_id_categorical"] = {"auc": roc_auc_score(y_valid, pred), "logloss": log_loss(y_valid, pred), "fit_time_s": time.time() - t0}
    log(f"CatBoost+rawID(범주) — AUC={results['catboost_raw_id_categorical']['auc']:.5f}  (baseline: {tuned['catboost']['final_auc']:.5f})")

    with open("raw_id_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("Saved raw_id_results.json")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
