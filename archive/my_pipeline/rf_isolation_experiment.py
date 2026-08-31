"""RandomForest 격차의 원인이 '알고리즘'인지 '피처셋'인지 분리 검증.

2x2:
  A) sklearn RandomForest + 팀원의 build_rf_exp01_features (그대로 복제)
  B) sklearn RandomForest + 우리 engineer_features (65피처, 지금까지 쓰던 것)

둘 다 팀원 RF와 같은 n_estimators=300 (meta.json에서 확인된 값), 나머지는 sklearn 기본값
(max_depth=None — 완전 성장 트리, 이게 아마 팀원도 썼을 가장 단순한 설정).
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from features import DROP_COLS, ENG_CAT_COLS, TARGET_COL, engineer_features

DATA_DIR = "../data"
RANDOM_STATE = 42
ID_COL = "row_id"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_rf_exp01_features(df):
    """teammate_submit/script.py의 build_rf_exp01_features 그대로 복제."""
    x = df.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in df.columns]).copy()
    x["count_state"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["two_strike"] = (x["strikes_before"] == 2).astype("int8")
    x["three_ball"] = (x["balls_before"] == 3).astype("int8")
    x["full_count"] = ((x["balls_before"] == 3) & (x["strikes_before"] == 2)).astype("int8")
    x["late_inning"] = (x["inning"] >= 7).astype("int8")
    x["score_abs_pitcher"] = x["score_diff_pitcher_team"].abs()
    x["close_game"] = (x["score_abs_pitcher"] <= 1).astype("int8")
    x["runner_scoring_position"] = ((x["runner_on_2b"] == 1) | (x["runner_on_3b"] == 1)).astype("int8")
    x["same_hand"] = (x["pitcher_hand"] == x["batter_hand"]).astype("int8")
    return x


def fit_predict_rf(X_train, X_valid, y_train, y_valid, cat_cols, label):
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # 범주형: train으로만 fit한 순서형 인코딩 (unseen -> -1)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train_cat = enc.fit_transform(X_train[cat_cols].astype(str))
    X_valid_cat = enc.transform(X_valid[cat_cols].astype(str))

    # 수치형: train 중앙값으로 결측 대치
    imp = SimpleImputer(strategy="median")
    X_train_num = imp.fit_transform(X_train[num_cols])
    X_valid_num = imp.transform(X_valid[num_cols])

    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_valid_final = np.hstack([X_valid_num, X_valid_cat])

    log(f"[{label}] fit 시작 (train={X_train_final.shape}, n_estimators=300, max_depth=None) ...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
    )
    model.fit(X_train_final, y_train)
    fit_time = time.time() - t0
    log(f"[{label}] fit 완료 ({fit_time:.1f}s)")

    pred = model.predict_proba(X_valid_final)[:, 1]
    auc = roc_auc_score(y_valid, pred)
    ll = log_loss(y_valid, pred)
    log(f"[{label}] AUC={auc:.5f}  LogLoss={ll:.5f}")
    return {"auc": auc, "logloss": ll, "fit_time_s": fit_time}


def main():
    log("Loading train.csv ...")
    train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    train_mask = train_raw["season"] < 2024
    y = train_raw[TARGET_COL]
    y_train, y_valid = y[train_mask], y[~train_mask]

    results = {}

    # ---- A) 팀원 피처셋 그대로 복제 ----
    X_a = build_rf_exp01_features(train_raw)
    cat_cols_a = ["top_bottom", "game_type", "base_state", "count_state"]
    X_a_train, X_a_valid = X_a[train_mask], X_a[~train_mask]
    results["A_teammate_features"] = fit_predict_rf(X_a_train, X_a_valid, y_train, y_valid, cat_cols_a, "A:팀원피처+sklearnRF")

    # ---- B) 우리 피처셋 (65개, 지금까지 쓰던 engineer_features) ----
    train_eng = engineer_features(train_raw)
    feature_cols_b = [c for c in train_eng.columns if c not in DROP_COLS]
    X_b = train_eng[feature_cols_b]
    X_b_train, X_b_valid = X_b[train_mask], X_b[~train_mask]
    results["B_our_features"] = fit_predict_rf(X_b_train, X_b_valid, y_train, y_valid, ENG_CAT_COLS, "B:우리피처+sklearnRF")

    with open("rf_isolation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("Saved rf_isolation_results.json")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
