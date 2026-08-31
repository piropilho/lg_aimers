"""팀원 파이프라인의 add_context_from_state 이식 — 선수x상황 조건부 deviation 인코딩 효과 검증.

pitcher_id x count_code, pitcher_id x hand_matchup, batter_id x count_code 3개 조합에 대해
"이 선수가 이 상황에서 자기 평균 대비 얼마나 벗어나는지"를 인코딩해서 추가.
튜닝에서 찾은 best_params 그대로 쓰고, 이 피처 추가 전/후 AUC만 비교한다.
"""

import json
import time

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from features import DROP_COLS, ENG_CAT_COLS, TARGET_COL, add_context_deviation, build_matrix, engineer_features

DATA_DIR = "../data"
RANDOM_STATE = 42

# (feature_prefix, player_col, context_cols, prior_player, prior_context)
CONTEXT_SPECS = [
    ("pitcher_count", "pitcher_id", ["count_code"], 500, 50),
    ("pitcher_hand", "pitcher_id", ["hand_matchup"], 500, 100),
    ("batter_count", "batter_id", ["count_code"], 500, 50),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    with open("tuning_results.json", encoding="utf-8") as f:
        tuned = json.load(f)

    log("Loading train.csv ...")
    train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    train_eng = engineer_features(train_raw)
    train_mask = train_eng["season"] < 2024

    new_cols = []
    for prefix, player_col, context_cols, prior_player, prior_context in CONTEXT_SPECS:
        log(f"Computing context deviation: {player_col} x {context_cols} (prefix={prefix}) ...")
        dev, log_n, rel = add_context_deviation(
            train_eng, train_mask, TARGET_COL, player_col, context_cols,
            prior_player=prior_player, prior_context=prior_context,
        )
        train_eng[f"{prefix}_deviation"] = dev
        train_eng[f"{prefix}_log_n"] = log_n
        train_eng[f"{prefix}_reliability"] = rel
        new_cols += [f"{prefix}_deviation", f"{prefix}_log_n", f"{prefix}_reliability"]

    feature_cols = [c for c in train_eng.columns if c not in DROP_COLS]
    y = train_eng[TARGET_COL]
    log(f"features={len(feature_cols)} (context deviation {len(new_cols)}개 추가됨: {new_cols})")

    X_tree = build_matrix(train_eng, feature_cols, ENG_CAT_COLS, mode="tree")
    X_cb = build_matrix(train_eng, feature_cols, ENG_CAT_COLS, mode="cb")
    X_train_tree, X_valid_tree = X_tree[train_mask], X_tree[~train_mask]
    X_train_cb, X_valid_cb = X_cb[train_mask], X_cb[~train_mask]
    y_train, y_valid = y[train_mask], y[~train_mask]

    results = {}

    # ---------------- XGBoost ----------------
    log("Training XGBoost (tuned params + context deviation) ...")
    t0 = time.time()
    model = xgb.XGBClassifier(
        **tuned["xgboost"]["best_params"], n_estimators=2000,
        tree_method="hist", enable_categorical=True,
        eval_metric="logloss", early_stopping_rounds=50,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_train_tree, y_train, eval_set=[(X_valid_tree, y_valid)], verbose=False)
    pred = model.predict_proba(X_valid_tree)[:, 1]
    results["xgboost"] = {"auc": roc_auc_score(y_valid, pred), "logloss": log_loss(y_valid, pred), "fit_time_s": time.time() - t0}
    log(f"XGBoost+CTX — AUC={results['xgboost']['auc']:.5f}  (baseline tuned: {tuned['xgboost']['final_auc']:.5f})")
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    log(f"  CTX 피처 importance 순위: {[f'{c}(#{list(imp.index).index(c)+1})' for c in new_cols]}")

    # ---------------- LightGBM ----------------
    log("Training LightGBM (tuned params + context deviation) ...")
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
    results["lightgbm"] = {"auc": roc_auc_score(y_valid, pred), "logloss": log_loss(y_valid, pred), "fit_time_s": time.time() - t0}
    log(f"LightGBM+CTX — AUC={results['lightgbm']['auc']:.5f}  (baseline tuned: {tuned['lightgbm']['final_auc']:.5f})")

    # ---------------- CatBoost ----------------
    log("Training CatBoost (tuned params + context deviation) ...")
    t0 = time.time()
    model = CatBoostClassifier(
        **tuned["catboost"]["best_params"], iterations=3000,
        cat_features=ENG_CAT_COLS, eval_metric="Logloss",
        random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=50,
    )
    model.fit(X_train_cb, y_train, eval_set=(X_valid_cb, y_valid))
    pred = model.predict_proba(X_valid_cb)[:, 1]
    results["catboost"] = {"auc": roc_auc_score(y_valid, pred), "logloss": log_loss(y_valid, pred), "fit_time_s": time.time() - t0}
    log(f"CatBoost+CTX — AUC={results['catboost']['auc']:.5f}  (baseline tuned: {tuned['catboost']['final_auc']:.5f})")

    imp_cb = pd.Series(model.get_feature_importance(), index=feature_cols).sort_values(ascending=False)
    log(f"  CatBoost 전체 top15: {list(imp_cb.head(15).index)}")
    log(f"  CTX 피처 importance 순위: {[f'{c}(#{list(imp_cb.index).index(c)+1})' for c in new_cols]}")

    with open("context_deviation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("Saved context_deviation_results.json")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
