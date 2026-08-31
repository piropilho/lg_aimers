"""공통 피처 엔지니어링 — 02_baseline_models.ipynb에서 검증된 로직을 재사용 가능하게 분리."""

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
]
ENG_CAT_COLS = CAT_COLS + ["count_code", "hand_matchup"]
DROP_COLS = [ID_COL, TARGET_COL, "pitcher_id", "batter_id"]


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


def add_target_encoding(df, train_mask, target_col, group_col, n_splits=5, smoothing=200, seed=42):
    """선수 단위 target encoding (shrinkage 적용, 시간/자기참조 누수 방지).

    - train 구간(train_mask=True): K-fold out-of-fold — 자기 자신의 행은 자기 인코딩에 안 쓰임
    - valid/test 구간(train_mask=False): train 전체로 계산한 통계를 그대로 적용 (미래→과거 방향이라 안전)
    - smoothing: 표본이 작은 선수를 global_mean 쪽으로 당기는 강도 (표본수 median 대비 낮게 잡음)

    Returns: (encoded_rate, log1p(count)) — 둘 다 df.index 기준 Series
    """
    from sklearn.model_selection import KFold  # 실험용 함수라 지연 import — 배포 스크립트는 sklearn 불필요

    global_mean = df.loc[train_mask, target_col].mean()
    encoded = pd.Series(np.nan, index=df.index, dtype="float64")
    count_feat = pd.Series(np.nan, index=df.index, dtype="float64")

    train_idx = df.index[train_mask].to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_pos, val_pos in kf.split(train_idx):
        fit_idx, val_idx = train_idx[fit_pos], train_idx[val_pos]
        stats = df.loc[fit_idx].groupby(group_col)[target_col].agg(["sum", "count"])
        mapped_sum = df.loc[val_idx, group_col].map(stats["sum"]).fillna(0)
        mapped_count = df.loc[val_idx, group_col].map(stats["count"]).fillna(0)
        encoded.loc[val_idx] = ((mapped_sum + smoothing * global_mean) / (mapped_count + smoothing)).to_numpy()
        count_feat.loc[val_idx] = mapped_count.to_numpy()

    other_idx = df.index[~train_mask].to_numpy()
    full_stats = df.loc[train_mask].groupby(group_col)[target_col].agg(["sum", "count"])
    mapped_sum = df.loc[other_idx, group_col].map(full_stats["sum"]).fillna(0)
    mapped_count = df.loc[other_idx, group_col].map(full_stats["count"]).fillna(0)
    encoded.loc[other_idx] = ((mapped_sum + smoothing * global_mean) / (mapped_count + smoothing)).to_numpy()
    count_feat.loc[other_idx] = mapped_count.to_numpy()

    return encoded, np.log1p(count_feat)


def build_matrix(df, feature_cols, cat_cols, mode="tree"):
    """mode='tree' -> xgboost/lightgbm용 (category dtype), mode='cb' -> catboost용 (string)."""
    x = df[feature_cols].copy()
    for c in cat_cols:
        x[c] = x[c].astype(str)
        if mode == "tree":
            x[c] = x[c].astype("category")
    return x
