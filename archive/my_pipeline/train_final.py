"""최종 제출용 CatBoost 모델 학습.

절차:
  1. train.csv 전체(2019-2024)에서 랜덤 5%를 early-stopping 확인용으로만 떼어
     최적 iteration 수를 구함 (일반화 성능 검증은 이미 시즌 홀드아웃으로 끝났으므로,
     여기서는 트리 개수 캘리브레이션이 목적)
  2. 그 iteration 수 + 약간의 버퍼로 100% 데이터에 최종 재학습 (데이터 최대 활용)
  3. submit/model/catboost_final.cbm 으로 저장 (pickle 대신 CatBoost 네이티브 포맷 —
     버전 호환성 문제 없이 이식 가능)
"""

import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from features import DROP_COLS, ENG_CAT_COLS, TARGET_COL, build_matrix, engineer_features

DATA_DIR = "../data"
RANDOM_STATE = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    with open("tuning_results.json", encoding="utf-8") as f:
        best_params = json.load(f)["catboost"]["best_params"]
    log(f"best_params = {best_params}")

    log("Loading train.csv (전체 2019-2024) ...")
    train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    train_eng = engineer_features(train_raw)
    feature_cols = [c for c in train_eng.columns if c not in DROP_COLS]
    y = train_eng[TARGET_COL]
    X = build_matrix(train_eng, feature_cols, ENG_CAT_COLS, mode="cb")
    log(f"전체 데이터: {X.shape}")

    # ---- 1단계: iteration 수 캘리브레이션 (랜덤 5% 홀드아웃) ----
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=0.05, random_state=RANDOM_STATE, stratify=y
    )
    log(f"캘리브레이션용 분할: fit={X_fit.shape}  cal={X_cal.shape}")

    cal_model = CatBoostClassifier(
        **best_params, iterations=3000,
        cat_features=ENG_CAT_COLS, eval_metric="Logloss",
        random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=50,
    )
    t0 = time.time()
    cal_model.fit(X_fit, y_fit, eval_set=(X_cal, y_cal))
    best_iter = cal_model.get_best_iteration()
    final_iterations = int(best_iter * 1.05) + 1  # 전체 데이터가 더 크니 약간의 버퍼
    log(f"캘리브레이션 완료 ({time.time()-t0:.1f}s) — best_iteration={best_iter} -> 최종 iterations={final_iterations}")

    cal_pred = cal_model.predict_proba(X_cal)[:, 1]
    log(f"  (참고) 5% 홀드아웃 AUC={roc_auc_score(y_cal, cal_pred):.5f}  LogLoss={log_loss(y_cal, cal_pred):.5f}")

    # ---- 2단계: 전체 데이터로 최종 재학습 ----
    log(f"전체 데이터로 최종 재학습 (iterations={final_iterations}, eval_set 없음) ...")
    final_model = CatBoostClassifier(
        **best_params, iterations=final_iterations,
        cat_features=ENG_CAT_COLS,
        random_seed=RANDOM_STATE, verbose=False,
    )
    t0 = time.time()
    final_model.fit(X, y)
    log(f"최종 학습 완료 ({time.time()-t0:.1f}s)")

    import os
    os.makedirs("submit/model", exist_ok=True)
    final_model.save_model("submit/model/catboost_final.cbm")
    log("저장: submit/model/catboost_final.cbm")

    with open("submit/model/feature_cols.json", "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "cat_cols": ENG_CAT_COLS}, f, indent=2, ensure_ascii=False)
    log("저장: submit/model/feature_cols.json")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
