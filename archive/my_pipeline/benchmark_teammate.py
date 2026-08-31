"""팀원 파이프라인(teammate_submit)을 우리와 동일한 2024 시즌 홀드아웃으로 실측 벤치마킹.

teammate_submit/model/catboost_original_bundle.pkl 안에 이미
'split': 'train=2019-2023, validation=2024' 메타데이터가 남아있어 우리 실험과
정확히 같은 분할이라는 게 확인됨 — 그대로 재현해서 AUC/LogLoss/Brier를 우리 쪽과 비교한다.
"""

import importlib.util
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

DATA_DIR = "../data"
TARGET_COL = "control_success"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_teammate_module():
    spec = importlib.util.spec_from_file_location("teammate_script", "../teammate_submit/script.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    tm = load_teammate_module()

    log("Loading train.csv, 2024 시즌만 추출 ...")
    train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
    valid = train_raw[train_raw["season"] == 2024].reset_index(drop=True)
    y_true = valid[TARGET_COL]
    log(f"valid shape={valid.shape}  성공률={y_true.mean():.4f}")

    log("모델 번들 로드 ...")
    champion_bundle = joblib.load(tm.find_path(Path("model/catboost_original_bundle.pkl")))
    a1_bundle = joblib.load(tm.find_path(Path("model/a1_catboost_bundle.pkl")))
    a2_bundle = joblib.load(tm.find_path(Path("model/a2_aux_bundle.pkl")))
    d1_bundle = joblib.load(tm.find_path(Path("model/d1_student_bundle.pkl")))
    rf_meta, rf_trees = tm.load_portable_rf()

    def report(name, pred):
        auc = roc_auc_score(y_true, pred)
        ll = log_loss(y_true, pred)
        brier = brier_score_loss(y_true, pred)
        log(f"{name:30s} AUC={auc:.5f}  LogLoss={ll:.5f}  Brier={brier:.5f}")
        return auc, ll, brier

    log("추론 시작 ...")
    t0 = time.time()

    p_rf = tm.predict_portable_rf(tm.build_rf_exp01_features(valid), rf_meta, rf_trees)
    report("RF (portable) 단독", p_rf)

    p_champion_cb = tm.predict_catboost(valid, champion_bundle)
    report("CatBoost champion 단독", p_champion_cb)

    p_champion = tm.CATBOOST_WEIGHT * p_champion_cb + tm.RF_WEIGHT * p_rf
    report("Champion (CB95+RF5)", p_champion)

    extra = pd.concat([
        tm.a1_features(valid, a1_bundle["aux_pitch"]),
        tm.a2_features(valid, a2_bundle),
    ], axis=1)
    p_student = tm.predict_student(valid, d1_bundle, extra)
    report("Student (d1) 단독", p_student)

    pred_final = (1 - tm.D1_BLEND_WEIGHT) * p_champion + tm.D1_BLEND_WEIGHT * p_student
    auc_final, ll_final, brier_final = report("최종 블렌드 (champion80+student20)", pred_final)

    log(f"추론 완료 ({time.time()-t0:.1f}s)")

    log("")
    log(f"팀원 번들 자체 기록 metadata: {champion_bundle['validation']}")
    log(f"  -> 우리가 방금 실측한 champion 단독 Brier와 비교해볼 것")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
