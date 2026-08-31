"""옵션 2: 튜닝된 XGBoost/LightGBM/CatBoost(TE 없음) 예측 블렌딩 효과 검증."""

import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from features import DROP_COLS, ENG_CAT_COLS, TARGET_COL, build_matrix, engineer_features

DATA_DIR = "../data"
RANDOM_STATE = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    with open("tuning_results.json", encoding="utf-8") as f:
        tuned = json.load(f)

    log("Loading train.csv ...")
    train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    train_eng = engineer_features(train_raw)  # TE 없음 — 저번에 성능 저하 확인됨
    train_mask = train_eng["season"] < 2024

    feature_cols = [c for c in train_eng.columns if c not in DROP_COLS]
    y = train_eng[TARGET_COL]

    X_tree = build_matrix(train_eng, feature_cols, ENG_CAT_COLS, mode="tree")
    X_cb = build_matrix(train_eng, feature_cols, ENG_CAT_COLS, mode="cb")
    X_train_tree, X_valid_tree = X_tree[train_mask], X_tree[~train_mask]
    X_train_cb, X_valid_cb = X_cb[train_mask], X_cb[~train_mask]
    y_train, y_valid = y[train_mask], y[~train_mask]

    preds = {}

    log("Training XGBoost (tuned) ...")
    model = xgb.XGBClassifier(
        **tuned["xgboost"]["best_params"], n_estimators=2000,
        tree_method="hist", enable_categorical=True,
        eval_metric="logloss", early_stopping_rounds=50,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train_tree, y_train, eval_set=[(X_valid_tree, y_valid)], verbose=False)
    preds["xgboost"] = model.predict_proba(X_valid_tree)[:, 1]
    log(f"  XGBoost AUC={roc_auc_score(y_valid, preds['xgboost']):.5f}")

    log("Training LightGBM (tuned) ...")
    model = lgb.LGBMClassifier(
        **tuned["lightgbm"]["best_params"], n_estimators=2000,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    model.fit(
        X_train_tree, y_train,
        eval_set=[(X_valid_tree, y_valid)], eval_metric="logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    preds["lightgbm"] = model.predict_proba(X_valid_tree)[:, 1]
    log(f"  LightGBM AUC={roc_auc_score(y_valid, preds['lightgbm']):.5f}")

    log("Training CatBoost (tuned) ...")
    model = CatBoostClassifier(
        **tuned["catboost"]["best_params"], iterations=3000,
        cat_features=ENG_CAT_COLS, eval_metric="Logloss",
        random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=50,
    )
    model.fit(X_train_cb, y_train, eval_set=(X_valid_cb, y_valid))
    preds["catboost"] = model.predict_proba(X_valid_cb)[:, 1]
    log(f"  CatBoost AUC={roc_auc_score(y_valid, preds['catboost']):.5f}")

    # 예측끼리 얼마나 상관되어 있는지 (다양성 확인 — 너무 비슷하면 블렌딩 이득 적음)
    pred_df = pd.DataFrame(preds)
    log("예측 간 상관관계:\n" + pred_df.corr().to_string())

    results = {"individual": {k: {"auc": roc_auc_score(y_valid, v), "logloss": log_loss(y_valid, v)} for k, v in preds.items()}}

    # 1) 단순 평균 (편향 없음 — 정직한 숫자)
    avg_pred = pred_df.mean(axis=1).to_numpy()
    results["equal_average"] = {"auc": roc_auc_score(y_valid, avg_pred), "logloss": log_loss(y_valid, avg_pred)}
    log(f"단순 평균 블렌드 — AUC={results['equal_average']['auc']:.5f}  LogLoss={results['equal_average']['logloss']:.5f}")

    # 2) 가중치 그리드서치 (validation에 최적화 — 낙관적 상한선, 참고용)
    best = {"auc": -1, "weights": None}
    step = 0.05
    grid = np.arange(0, 1.0001, step)
    for w_xgb in grid:
        for w_lgb in grid:
            w_cb = 1 - w_xgb - w_lgb
            if w_cb < -1e-9 or w_cb > 1 + 1e-9:
                continue
            w_cb = max(w_cb, 0.0)
            blend = w_xgb * preds["xgboost"] + w_lgb * preds["lightgbm"] + w_cb * preds["catboost"]
            auc = roc_auc_score(y_valid, blend)
            if auc > best["auc"]:
                best = {"auc": auc, "weights": {"xgboost": round(float(w_xgb), 2), "lightgbm": round(float(w_lgb), 2), "catboost": round(float(w_cb), 2)}}
    best_blend = (
        best["weights"]["xgboost"] * preds["xgboost"]
        + best["weights"]["lightgbm"] * preds["lightgbm"]
        + best["weights"]["catboost"] * preds["catboost"]
    )
    results["optimized_weights_UPPER_BOUND"] = {
        "weights": best["weights"],
        "auc": roc_auc_score(y_valid, best_blend),
        "logloss": log_loss(y_valid, best_blend),
        "note": "validation set에서 직접 탐색한 가중치라 낙관 편향 있음 (참고용 상한선)",
    }
    log(f"가중치 최적화 블렌드 (참고용) — weights={best['weights']}  AUC={results['optimized_weights_UPPER_BOUND']['auc']:.5f}")

    with open("blend_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("Saved blend_results.json")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
