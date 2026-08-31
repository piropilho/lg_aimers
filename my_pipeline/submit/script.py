# script.py — CatBoost 최종 제출 파이프라인
#
# model/catboost_final.cbm 을 불러와 data/test.csv에 대해 추론하고
# output/submission.csv를 생성한다. 피처 엔지니어링 로직은 my_pipeline의
# features.py / 02_baseline_models.ipynb에서 검증된 것을 그대로 고정해 이식했다.

import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ID_COL = "row_id"
TARGET_COL = "control_success"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_path(relative):
    """실행 위치(script 폴더 기준 / 현재 작업 폴더 기준 / open/ 하위)를 순서대로 탐색."""
    candidates = [
        os.path.join(BASE_DIR, relative),
        os.path.join(os.getcwd(), relative),
        os.path.join(BASE_DIR, "open", relative),
        os.path.join(os.getcwd(), "open", relative),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"missing required file {relative}; checked {candidates}")


# =======================
# 피처 엔지니어링 (학습 때와 동일 — my_pipeline/features.py 고정본)
# =======================

def engineer_features(df):
    x = df.copy()
    x["count_code"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["full_count"] = ((x["balls_before"] == 3) & (x["strikes_before"] == 2)).astype("int8")
    x["three_ball"] = (x["balls_before"] == 3).astype("int8")
    x["two_strike"] = (x["strikes_before"] == 2).astype("int8")

    x["same_hand"] = (x["pitcher_hand"] == x["batter_hand"]).astype("int8")
    x["hand_matchup"] = x["pitcher_hand"].astype(str) + "-" + x["batter_hand"].astype(str)

    x["score_abs_pitcher"] = x["score_diff_pitcher_team"].abs()
    x["blowout"] = (x["score_abs_pitcher"] >= 5).astype("int8")
    x["close_game"] = (x["score_abs_pitcher"] <= 1).astype("int8")

    x["runner_scoring_pos"] = x["runner_on_2b"] + x["runner_on_3b"]
    x["late_inning"] = (x["inning"] >= 7).astype("int8")

    x["form_delta1"] = x["asof_pitcher_prev1_game_success_rate"] - x["asof_pitcher_success_rate"]
    x["form_delta3"] = x["asof_pitcher_prev3_game_success_rate"] - x["asof_pitcher_success_rate"]
    x["form_delta5"] = x["asof_pitcher_prev5_game_success_rate"] - x["asof_pitcher_success_rate"]
    x["form_declining"] = (x["form_delta1"] < -0.05).astype("int8")

    mix = x[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]].clip(1e-6, 1).fillna(1 / 3)
    x["pitchmix_entropy"] = -(mix * np.log(mix)).sum(axis=1)
    x["pitchmix_max"] = mix.max(axis=1)

    x["command_quality"] = x["asof_pitcher_success_rate"] - x["asof_pitcher_reverse_rate"] - x["asof_pitcher_middle_rate"]
    x["strike_ball_gap"] = x["asof_pitcher_strike_rate"] - x["asof_pitcher_ball_rate"]

    x["stress_score"] = x["full_count"] + x["blowout"] + x["form_declining"]
    return x


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}")
    return df


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    MODEL_PATH = find_path(os.path.join("model", "catboost_final.cbm"))
    FEATURE_META_PATH = find_path(os.path.join("model", "feature_cols.json"))
    TEST_PATH = find_path(os.path.join("data", "test.csv"))
    SAMPLE_SUB_PATH = find_path(os.path.join("data", "sample_submission.csv"))
    OUT_PATH = os.path.join(BASE_DIR, "output", "submission.csv")

    print("Load model...")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    with open(FEATURE_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols, cat_cols = meta["feature_cols"], meta["cat_cols"]
    print(f" OK. n_features={len(feature_cols)}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    test_eng = engineer_features(test)
    X = test_eng[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    print(f" features={X.shape[1]}")

    print("Inference model...")
    preds = model.predict_proba(X)[:, 1] if len(X) else []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"✅ Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
