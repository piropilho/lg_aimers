"""최종 submit.zip 블렌드에 대한 temperature 파라미터 확정 fit.

- teammate_submit main()과 동일하게 2024 블렌드 예측 생성
- 전체 2024에서 Brier 최소화하는 T 산출 (배포용 고정값)
- p_cal = sigmoid(logit(p)/T)
"""
import importlib.util
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

BASE = Path(__file__).resolve().parent.parent  # repo root
TARGET = "control_success"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_tm():
    spec = importlib.util.spec_from_file_location("tm", BASE / "teammate_submit/script.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def best_T(p, y):
    z = logit(p)
    grid = np.arange(0.60, 1.201, 0.001)
    briers = [brier_score_loss(y, sigmoid(z / T)) for T in grid]
    T = float(grid[int(np.argmin(briers))])
    return T, min(briers)


def main():
    tm = load_tm()
    raw = pd.read_csv(BASE / "data/train.csv", encoding="utf-8-sig")
    va = raw[raw.season == 2024].reset_index(drop=True)
    y = va[TARGET].to_numpy()
    log(f"2024 rows={len(va)} rate={y.mean():.4f}")

    champ = joblib.load(tm.find_path(Path("model/catboost_original_bundle.pkl")))
    a1 = joblib.load(tm.find_path(Path("model/a1_catboost_bundle.pkl")))
    a2 = joblib.load(tm.find_path(Path("model/a2_aux_bundle.pkl")))
    d1 = joblib.load(tm.find_path(Path("model/d1_student_bundle.pkl")))
    rf_meta, rf_trees = tm.load_portable_rf()

    log("inference ...")
    p_rf = tm.predict_portable_rf(tm.build_rf_exp01_features(va), rf_meta, rf_trees)
    p_cb = tm.predict_catboost(va, champ)
    extra = pd.concat([tm.a1_features(va, a1["aux_pitch"]), tm.a2_features(va, a2)], axis=1)
    p_stu = tm.predict_student(va, d1, extra)
    p_champion = tm.CATBOOST_WEIGHT * p_cb + tm.RF_WEIGHT * p_rf
    p_blend = (1 - tm.D1_BLEND_WEIGHT) * p_champion + tm.D1_BLEND_WEIGHT * p_stu

    b0 = brier_score_loss(y, p_blend)
    T, bT = best_T(p_blend, y)
    p_cal = sigmoid(logit(p_blend) / T)
    log("")
    log(f"blend raw     Brier={b0:.6f}  LogLoss={log_loss(y, p_blend):.5f}  AUC={roc_auc_score(y, p_blend):.5f}  mean={p_blend.mean():.4f}")
    log(f"blend T={T:.3f}  Brier={bT:.6f}  LogLoss={log_loss(y, p_cal):.5f}  AUC={roc_auc_score(y, p_cal):.5f}  mean={p_cal.mean():.4f}")
    log(f"Δ Brier = {bT - b0:+.6f}   (BSS×1e5 Δ ≈ {-(bT - b0) / 0.249807 * 1e5:+.1f})")
    log("")
    log(f">>> DEPLOY_TEMPERATURE = {T:.3f}")


if __name__ == "__main__":
    main()
