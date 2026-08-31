"""최종 블렌드 캘리브레이션 검토.

질문: submit.zip의 최종 예측을 2024로 캘리브레이션하면 Brier가 깎이나?
프로토콜:
  1. teammate_submit 파이프라인 그대로 2024에 추론 -> raw 블렌드 예측
  2. raw 블렌드의 캘리브레이션 상태 진단 (Murphy 분해 reliability항, ECE, decile 신뢰도표)
  3. 캘리브레이터 5종을 2024 내부 2-fold CV로 정직 평가 (fit A -> eval B)
     - identity(무보정) / shift(로짓 1param) / temp(로짓 스케일 1param) / platt(2param) / isotonic
  4. full-2024 fit&eval(낙관 상한)도 같이 찍어 과적합 위험 확인
  5. champion 단독 / RF가중 0 변형도 같이

주의: portable RF는 full_train(2024 포함) -> 블렌드의 RF 4%가 in-sample.
      2025 실제보다 약간 잘 맞는 쪽으로 편향. RF weight 0 변형으로 대조.
"""
import importlib.util
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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


def murphy(y, p, bins=15):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = np.quantile(p, np.linspace(0, 1, bins + 1)); e[0] -= 1e-9; e[-1] += 1e-9
    idx = np.digitize(p, e) - 1
    obar = y.mean(); rel = res = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        nk = m.sum()
        rel += nk * (p[m].mean() - y[m].mean()) ** 2
        res += nk * (y[m].mean() - obar) ** 2
    n = len(y)
    return rel / n, res / n, obar * (1 - obar)


def ece(y, p, bins=15):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = np.quantile(p, np.linspace(0, 1, bins + 1)); e[0] -= 1e-9; e[-1] += 1e-9
    idx = np.digitize(p, e) - 1
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return tot / len(y)


def reliability_table(y, p, bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    e = np.quantile(p, np.linspace(0, 1, bins + 1)); e[0] -= 1e-9; e[-1] += 1e-9
    idx = np.digitize(p, e) - 1
    out = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            out.append((b, 0, np.nan, np.nan)); continue
        out.append((b, int(m.sum()), p[m].mean(), y[m].mean()))
    return out


def fit_shift(p, y):
    z = logit(p); best = (1e9, 0.0)
    for s in np.linspace(-0.6, 0.6, 241):
        b = brier_score_loss(y, sigmoid(z + s))
        if b < best[0]:
            best = (b, s)
    s = best[1]
    return (lambda q: sigmoid(logit(q) + s)), round(s, 4)


def fit_temp(p, y):
    z = logit(p); best = (1e9, 1.0)
    for T in np.linspace(0.4, 3.0, 261):
        b = brier_score_loss(y, sigmoid(z / T))
        if b < best[0]:
            best = (b, T)
    T = best[1]
    return (lambda q: sigmoid(logit(q) / T)), round(T, 4)


def fit_platt(p, y):
    lr = LogisticRegression(C=1e6)
    lr.fit(logit(p).reshape(-1, 1), y)
    a, b = lr.coef_[0, 0], lr.intercept_[0]
    return (lambda q: sigmoid(a * logit(q) + b)), (round(a, 4), round(b, 4))


def fit_iso(p, y):
    ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5)
    ir.fit(p, y)
    return (lambda q: np.clip(ir.predict(q), 1e-5, 1 - 1e-5)), "monotone"


METHODS = {"identity": None, "shift(1p)": fit_shift, "temp(1p)": fit_temp,
           "platt(2p)": fit_platt, "isotonic": fit_iso}


def diagnose(tag, y, p):
    b = brier_score_loss(y, p); ll = log_loss(y, p); auc = roc_auc_score(y, p)
    rel, res, unc = murphy(y, p); e = ece(y, p)
    log(f"{tag:20s} Brier={b:.6f}  LogLoss={ll:.5f}  AUC={auc:.5f}  reliability={rel:.3e}  ECE={e:.4f}  mean_pred={p.mean():.4f}")
    return b


def cv_eval(y, p):
    rng = np.random.RandomState(0)
    o = rng.permutation(len(y)); h = len(y) // 2
    folds = [(o[:h], o[h:]), (o[h:], o[:h])]
    base = np.mean([brier_score_loss(y[te], p[te]) for _, te in folds])
    for name, fitter in METHODS.items():
        if fitter is None:
            continue
        bs, ps = [], []
        for tr, te in folds:
            f, par = fitter(p[tr], y[tr])
            bs.append(brier_score_loss(y[te], np.asarray(f(p[te]), float)))
            ps.append(par)
        m = np.mean(bs)
        # full-fit optimistic bound
        f_full, par_full = fitter(p, y)
        b_full = brier_score_loss(y, np.asarray(f_full(p), float))
        log(f"  {name:12s} CV_Brier={m:.6f}  Δ={m - base:+.6f}   (full-fit bound={b_full:.6f}, Δ={b_full - base:+.6f})  params={ps}")
    log(f"  {'identity':12s} CV_Brier={base:.6f}   (reference)")


def main():
    tm = load_tm()
    log("load train.csv, 2024 ...")
    raw = pd.read_csv(BASE / "data/train.csv", encoding="utf-8-sig")
    va = raw[raw.season == 2024].reset_index(drop=True)
    y = va[TARGET].to_numpy()
    log(f"2024 rows={len(va)}  obs_rate={y.mean():.4f}  const_Brier={y.mean() * (1 - y.mean()):.6f}")

    champ = joblib.load(tm.find_path(Path("model/catboost_original_bundle.pkl")))
    a1 = joblib.load(tm.find_path(Path("model/a1_catboost_bundle.pkl")))
    a2 = joblib.load(tm.find_path(Path("model/a2_aux_bundle.pkl")))
    d1 = joblib.load(tm.find_path(Path("model/d1_student_bundle.pkl")))
    rf_meta, rf_trees = tm.load_portable_rf()

    log("inference ...")
    t0 = time.time()
    p_rf = tm.predict_portable_rf(tm.build_rf_exp01_features(va), rf_meta, rf_trees)
    p_cb = tm.predict_catboost(va, champ)
    extra = pd.concat([tm.a1_features(va, a1["aux_pitch"]), tm.a2_features(va, a2)], axis=1)
    p_stu = tm.predict_student(va, d1, extra)
    log(f"inference done {time.time() - t0:.1f}s")

    p_champ = tm.CATBOOST_WEIGHT * p_cb + tm.RF_WEIGHT * p_rf
    p_blend = (1 - tm.D1_BLEND_WEIGHT) * p_champ + tm.D1_BLEND_WEIGHT * p_stu
    p_blend_norf = (1 - tm.D1_BLEND_WEIGHT) * p_cb + tm.D1_BLEND_WEIGHT * p_stu

    log("")
    log("=== raw diagnostics (2024, full) ===")
    diagnose("blend (submit.zip)", y, p_blend)
    diagnose("champion CB alone", y, p_cb)
    diagnose("champion+RF5", y, p_champ)
    diagnose("blend, RF weight 0", y, p_blend_norf)

    log("")
    log("=== blend decile reliability (n, pred, obs, gap) ===")
    for b, n, pm, om in reliability_table(y, p_blend, 10):
        log(f"  d{b}  n={n:6d}  pred={pm:.4f}  obs={om:.4f}  gap={pm - om:+.4f}")

    log("")
    log("=== calibrators, honest 2-fold CV within 2024 ===")
    log("[submit.zip blend]")
    cv_eval(y, p_blend)
    log("[champion CB alone]")
    cv_eval(y, p_cb)
    log("[blend, RF weight 0]")
    cv_eval(y, p_blend_norf)

    log("ALL_DONE")


if __name__ == "__main__":
    main()
