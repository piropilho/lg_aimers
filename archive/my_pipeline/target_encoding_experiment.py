"""옵션 1: pitcher_id/batter_id target encoding(shrinkage) 추가 효과 검증.

튜닝에서 찾은 best_params를 그대로 쓰고, target encoding 피처 추가 전/후 AUC만 비교한다.
"""

import json
import time

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from features import DROP_COLS, ENG_CAT_COLS, TARGET_COL, add_target_encoding, build_matrix, engineer_features

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

    log("Computing target encoding for pitcher_id / batter_id ...")
    te_pitcher_rate, te_pitcher_log_n = add_target_encoding(
        train_eng, train_mask, TARGET_COL, "pitcher_id", smoothing=200
    )
    te_batter_rate, te_batter_log_n = add_target_encoding(
        train_eng, train_mask, TARGET_COL, "batter_id", smoothing=200
    )
    train_eng["te_pitcher_rate"] = te_pitcher_rate
    train_eng["te_pitcher_log_n"] = te_pitcher_log_n
    train_eng["te_batter_rate"] = te_batter_rate
    train_eng["te_batter_log_n"] = te_batter_log_n

    feature_cols = [c for c in train_eng.columns if c not in DROP_COLS]
    y = train_eng[TARGET_COL]
    log(f"features={len(feature_cols)} (target encoding 4개 추가됨)")

    X_tree = build_matrix(train_eng, feature_cols, ENG_CAT_COLS, mode="tree")
    X_cb = build_matrix(train_eng, feature_cols, ENG_CAT_COLS, mode="cb")
    X_train_tree, X_valid_tree = X_tree[train_mask], X_tree[~train_mask]
    X_train_cb, X_valid_cb = X_cb[train_mask], X_cb[~train_mask]
    y_train, y_valid = y[train_mask], y[~train_mask]

    results = {}

    # ---------------- XGBoost ----------------
    log("Training XGBoost (tuned params + TE) ...")
    t0 = time.time()
    model = xgb.XGBClassifier(
        **tuned["xgboost"]["best_params"], n_estimators=2000,
        tree_method="hist", enable_categorical=True,
        eval_metric="logloss", early_stopping_rounds=50,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train_tree, y_train, eval_set=[(X_valid_tree, y_valid)], verbose=False)
    pred = model.predict_proba(X_valid_tree)[:, 1]
    results["xgboost"] = {
        "auc": roc_auc_score(y_valid, pred), "logloss": log_loss(y_valid, pred),
        "fit_time_s": time.time() - t0,
    }
    log(f"XGBoost+TE — AUC={results['xgboost']['auc']:.5f}  (baseline tuned: {tuned['xgboost']['final_auc']:.5f})")

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    log(f"  TE 피처 importance 순위: {[f'{c}(#{list(imp.index).index(c)+1})' for c in ['te_pitcher_rate','te_batter_rate','te_pitcher_log_n','te_batter_log_n']]}")

    # ---------------- LightGBM ----------------
    log("Training LightGBM (tuned params + TE) ...")
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
    results["lightgbm"] = {
        "auc": roc_auc_score(y_valid, pred), "logloss": log_loss(y_valid, pred),
        "fit_time_s": time.time() - t0,
    }
    log(f"LightGBM+TE — AUC={results['lightgbm']['auc']:.5f}  (baseline tuned: {tuned['lightgbm']['final_auc']:.5f})")

    # ---------------- CatBoost ----------------
    log("Training CatBoost (tuned params + TE) ...")
    t0 = time.time()
    model = CatBoostClassifier(
        **tuned["catboost"]["best_params"], iterations=3000,
        cat_features=ENG_CAT_COLS, eval_metric="Logloss",
        random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=50,
    )
    model.fit(X_train_cb, y_train, eval_set=(X_valid_cb, y_valid))
    pred = model.predict_proba(X_valid_cb)[:, 1]
    results["catboost"] = {
        "auc": roc_auc_score(y_valid, pred), "logloss": log_loss(y_valid, pred),
        "fit_time_s": time.time() - t0,
    }
    log(f"CatBoost+TE — AUC={results['catboost']['auc']:.5f}  (baseline tuned: {tuned['catboost']['final_auc']:.5f})")

    imp_cb = pd.Series(model.get_feature_importance(), index=feature_cols).sort_values(ascending=False)
    log(f"  CatBoost 전체 top10: {list(imp_cb.head(10).index)}")

    with open("target_encoding_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("Saved target_encoding_results.json")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
