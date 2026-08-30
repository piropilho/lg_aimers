"""D1 teacher-student blend candidate inference script."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
BASE_DIR = Path(__file__).resolve().parent
CATBOOST_WEIGHT = 0.95
RF_WEIGHT = 0.05
D1_BLEND_WEIGHT = 0.20
PITCH_GROUPS = ("fastball", "breaking", "offspeed")
PITCH_RATE_COLS = [f"asof_pitcher_{g}_rate" for g in PITCH_GROUPS]
A2_PHYSICAL_TARGET_COLS = ["rel_speed", "spin_rate", "movement_norm", "extension", "zone_speed"]
LOW_CATS = [
    "game_month", "game_dayofweek", "top_bottom", "game_type", "balls_before",
    "strikes_before", "outs_before", "base_state", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "count_code", "hand_matchup",
]


def find_path(relative):
    candidates = [BASE_DIR / relative, Path.cwd() / relative, BASE_DIR / "open" / relative, Path.cwd() / "open" / relative]
    if Path(relative).as_posix().startswith("model/"):
        candidates.extend([BASE_DIR / Path(relative).name, Path.cwd() / Path(relative).name])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"missing required file {relative}; checked {[str(p) for p in candidates]}")


def build_rf_exp01_features(df):
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


def load_portable_rf():
    with open(find_path(Path("model/rf_param001_full_train_portable_meta.json")), "r", encoding="utf-8") as f:
        meta = json.load(f)
    trees = np.load(find_path(Path("model/rf_param001_full_train_portable.npz")))
    return meta, trees


def transform_portable_rf(frame, meta):
    values = []
    for col in meta["cat_cols"]:
        mapping = {value: i for i, value in enumerate(meta["categories"][col])}
        values.append(frame[col].astype(str).map(mapping).fillna(-1).to_numpy(dtype="float64"))
    for col, median in zip(meta["num_cols"], meta["num_medians"]):
        numeric = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype="float64")
        values.append(np.where(np.isnan(numeric), median, numeric))
    return np.column_stack(values)


def predict_portable_rf(frame, meta, trees):
    x = transform_portable_rf(frame, meta)
    pred = np.zeros(x.shape[0], dtype="float64")
    for i in range(meta["n_estimators"]):
        children_left = trees[f"children_left_{i}"]
        children_right = trees[f"children_right_{i}"]
        feature = trees[f"feature_{i}"]
        threshold = trees[f"threshold_{i}"]
        value = trees[f"value_{i}"]
        node = np.zeros(x.shape[0], dtype=np.int32)
        active = children_left[node] != -1
        while np.any(active):
            row_index = np.where(active)[0]
            current_node = node[row_index]
            go_left = x[row_index, feature[current_node]] <= threshold[current_node]
            node[row_index] = np.where(go_left, children_left[current_node], children_right[current_node])
            active = children_left[node] != -1
        pred += value[node]
    return pred / meta["n_estimators"]


PLAYER_SPECS = {
    "pitcher": {"id": "pitcher_id", "n": "asof_pitcher_n", "rates": [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"], "target_rate": "asof_pitcher_success_rate"},
    "batter": {"id": "batter_id", "n": "asof_batter_n", "rates": [
        "asof_batter_success_rate", "asof_batter_middle_rate"], "target_rate": "asof_batter_success_rate"},
}


def build_row_features(df):
    x = df.drop(columns=[c for c in [ID_COL, TARGET_COL, "pitcher_id", "batter_id"] if c in df.columns]).copy()
    x["count_code"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["hand_matchup"] = x["pitcher_hand"].astype(str) + "-" + x["batter_hand"].astype(str)
    x["score_abs"] = x["score_diff_pitcher_team"].abs()
    x["is_tied"] = (x["score_diff_pitcher_team"] == 0).astype("int8")
    x["is_late"] = (x["inning"] >= 7).astype("int8")
    x["late_close"] = ((x["inning"] >= 7) & (x["score_abs"] <= 1)).astype("int8")
    x["count_pressure"] = (x["balls_before"] == 3).astype("int8")
    x["two_strike"] = (x["strikes_before"] == 2).astype("int8")
    x["runner_pressure"] = x["runner_on_2b"] + x["runner_on_3b"]
    x["win_balance"] = (x["home_win_expectancy"] - 50.0).abs()
    x["li_log"] = np.log1p(x["li"].clip(lower=0))
    pn = x["asof_pitcher_n"].astype("float64")
    bn = x["asof_batter_n"].astype("float64")
    pr = x["asof_pitcher_success_rate"]
    br = x["asof_batter_success_rate"]
    x["pitcher_log_n"] = np.log1p(pn)
    x["batter_log_n"] = np.log1p(bn)
    x["experience_ratio"] = x["pitcher_log_n"] - x["batter_log_n"]
    for prior in (50.0, 200.0, 1000.0):
        x[f"pitcher_success_shrunk_{int(prior)}"] = (pn * pr + prior * 0.5) / (pn + prior)
        x[f"batter_success_shrunk_{int(prior)}"] = (bn * br + prior * 0.5) / (bn + prior)
    recent_s = [x[f"asof_pitcher_prev{k}_game_success_rate"] for k in (1, 3, 5)]
    recent_m = [x[f"asof_pitcher_prev{k}_game_middle_rate"] for k in (1, 3, 5)]
    for k, v in zip((1, 3, 5), recent_s):
        x[f"form_delta_{k}"] = v - pr
    for k, v in zip((1, 3, 5), recent_m):
        x[f"middle_delta_{k}"] = v - x["asof_pitcher_middle_rate"]
    x["recent_success_mean"] = pd.concat(recent_s, axis=1).mean(axis=1)
    x["recent_success_slope"] = recent_s[0] - recent_s[2]
    x["recent_middle_mean"] = pd.concat(recent_m, axis=1).mean(axis=1)
    x["recent_middle_slope"] = recent_m[0] - recent_m[2]
    x["failure_component_sum"] = x["asof_pitcher_reverse_rate"] + x["asof_pitcher_middle_rate"] + x["asof_pitcher_ball_rate"]
    x["command_quality"] = pr - x["asof_pitcher_reverse_rate"] - x["asof_pitcher_middle_rate"]
    x["strike_ball_gap"] = x["asof_pitcher_strike_rate"] - x["asof_pitcher_ball_rate"]
    mix = x[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]].clip(1e-6, 1)
    x["pitchmix_entropy"] = -(mix * np.log(mix)).sum(axis=1)
    x["pitchmix_max"] = mix.max(axis=1)
    x["success_x_three_ball"] = pr * x["count_pressure"]
    x["success_x_two_strike"] = pr * x["two_strike"]
    x["success_x_late"] = pr * x["is_late"]
    x["success_x_li"] = pr * x["li_log"]
    for c in LOW_CATS:
        x[c] = x[c].fillna("__MISSING__").astype(str)
    return x


def _prior(raw, ends, id_col, season, second=False):
    eligible = ends[ends.season < season].sort_values([id_col, "season"], ascending=[True, False]).copy()
    eligible["_rank"] = eligible.groupby(id_col, observed=True).cumcount()
    table = eligible[eligible._rank == (1 if second else 0)].drop(columns="_rank").set_index(id_col)
    mapped = table.reindex(raw[id_col].to_numpy())
    mapped.index = raw.index
    return mapped


def add_season_features(x, raw, states):
    for label, spec in PLAYER_SPECS.items():
        f = pd.DataFrame(index=raw.index)
        for season in sorted(raw.season.unique()):
            mask = raw.season.eq(season)
            part = raw.loc[mask]
            p1 = _prior(part, states[label], spec["id"], int(season))
            p2 = _prior(part, states[label], spec["id"], int(season), True)
            total_n = part[spec["n"]].to_numpy(dtype="float64")
            for rate_col in spec["rates"]:
                success = rate_col == spec["target_rate"]
                short = "success" if success else rate_col.replace(f"asof_{label}_", "").replace("_rate", "")
                en = f"{label}_{short}_end_n"
                es = f"{label}_{short}_end_sum"
                n1 = p1[en].fillna(0).to_numpy()
                s1 = p1[es].fillna(0).to_numpy()
                n2 = p2[en].fillna(0).to_numpy()
                s2 = p2[es].fillna(0).to_numpy()
                cur_n = np.maximum(total_n - n1, 0)
                total_sum = total_n * part[rate_col].to_numpy()
                cur_sum = np.clip(total_sum - s1, 0, cur_n)
                raw_rate = np.divide(cur_sum, cur_n, out=np.full_like(cur_sum, 0.5), where=cur_n > 0)
                career = np.divide(s1, n1, out=np.full_like(s1, 0.5), where=n1 > 0)
                prev_n = np.maximum(n1 - n2, 0)
                prev_sum = np.clip(s1 - s2, 0, prev_n)
                prev_rate = np.divide(prev_sum, prev_n, out=career.copy(), where=prev_n > 0)
                f.loc[mask, f"season_{label}_{short}_n"] = np.log1p(cur_n)
                f.loc[mask, f"season_{label}_{short}_rate"] = raw_rate
                f.loc[mask, f"season_{label}_{short}_delta"] = raw_rate - part[rate_col].to_numpy()
                f.loc[mask, f"previous_{label}_{short}_n"] = np.log1p(prev_n)
                f.loc[mask, f"previous_{label}_{short}_rate"] = prev_rate
                f.loc[mask, f"previous_{label}_{short}_career_delta"] = prev_rate - career
                for strength in (25.0, 100.0, 500.0):
                    f.loc[mask, f"season_{label}_{short}_shrunk_{int(strength)}"] = (cur_sum + strength * career) / (cur_n + strength)
                    f.loc[mask, f"season_{label}_{short}_recent_shrunk_{int(strength)}"] = (cur_sum + strength * prev_rate) / (cur_n + strength)
            sn = np.expm1(f.loc[mask, f"season_{label}_success_n"].to_numpy())
            f.loc[mask, f"season_{label}_experience_fraction"] = sn / np.maximum(total_n, 1)
            f.loc[mask, f"season_{label}_cold"] = (sn < 25).astype("int8")
        x = pd.concat([x, f.astype("float32")], axis=1)
    return x


def add_context_from_state(x, raw, state):
    f = pd.DataFrame(index=raw.index)
    for prefix, e in state["entries"].items():
        player = e["player"]
        context = e["context"]
        prior = e["prior"]
        pidx = pd.Index(raw[player], name=player)
        ps = e["player_stats"]["sum"].reindex(pidx).to_numpy(dtype="float64")
        pc = e["player_stats"]["count"].reindex(pidx).to_numpy(dtype="float64")
        pr = (np.nan_to_num(ps) + 500 * e["global_rate"]) / (np.nan_to_num(pc) + 500)
        idx = pd.MultiIndex.from_frame(raw[[player] + context])
        gs = e["group_stats"]["sum"].reindex(idx).to_numpy(dtype="float64")
        gc = e["group_stats"]["count"].reindex(idx).to_numpy(dtype="float64")
        n = np.nan_to_num(gc)
        rate = (np.nan_to_num(gs) + prior * pr) / (n + prior)
        f[f"{prefix}_deviation"] = (rate - pr).astype("float32")
        f[f"{prefix}_log_n"] = np.log1p(n).astype("float32")
        f[f"{prefix}_reliability"] = (1 - np.exp(-n / prior)).astype("float32")
    return pd.concat([x, f], axis=1)


def add_pitchmix(x, raw, multiplier):
    idx = pd.MultiIndex.from_frame(raw[["balls_before", "strikes_before"]])
    mult = multiplier.reindex(idx).fillna(1).to_numpy(dtype="float64")
    prior = raw[PITCH_RATE_COLS].fillna(1 / 3).to_numpy(dtype="float64")
    expected = prior * mult
    expected /= np.maximum(expected.sum(axis=1, keepdims=True), 1e-9)
    f = pd.DataFrame(index=raw.index)
    for j, g in enumerate(PITCH_GROUPS):
        f[f"expected_{g}_at_count"] = expected[:, j]
        f[f"expected_{g}_count_delta"] = expected[:, j] - prior[:, j]
    f["expected_mix_entropy"] = -(np.clip(expected, 1e-7, 1) * np.log(np.clip(expected, 1e-7, 1))).sum(axis=1)
    f["expected_mix_max"] = expected.max(axis=1)
    return pd.concat([x, f.astype("float32")], axis=1)


def logit_shift(pred, shift):
    p = np.clip(np.asarray(pred, dtype="float64"), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + shift)))


def residual_correction(raw, maps):
    values = []
    for e in maps:
        keys = e["keys"]
        idx = pd.Index(raw[keys[0]], name=keys[0]) if len(keys) == 1 else pd.MultiIndex.from_frame(raw[keys])
        sums = e["stats"]["sum"].reindex(idx).to_numpy(dtype="float64")
        counts = e["stats"]["count"].reindex(idx).to_numpy(dtype="float64")
        values.append(np.nan_to_num(sums / (counts + e["prior"]), nan=0.0))
    return np.mean(values, axis=0)


def aux_pitch_base_frame(raw):
    x = raw.drop(columns=[c for c in [ID_COL, TARGET_COL] if c in raw.columns]).copy()
    x["count_state"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["hand_matchup"] = x["pitcher_hand"].astype(str) + "-" + x["batter_hand"].astype(str)
    x["score_abs"] = x["score_diff_pitcher_team"].abs()
    x["is_late"] = (x["inning"] >= 7).astype("int8")
    x["runner_pressure"] = x["runner_on_2b"] + x["runner_on_3b"]
    x["li_log"] = np.log1p(x["li"].clip(lower=0))
    mix = x[PITCH_RATE_COLS].clip(1e-6, 1).fillna(1 / 3)
    x["pitchmix_entropy"] = -(mix * np.log(mix)).sum(axis=1)
    x["pitchmix_max"] = mix.max(axis=1)
    cats = [
        "game_month", "game_dayofweek", "top_bottom", "game_type",
        "balls_before", "strikes_before", "outs_before", "base_state",
        "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
        "pitcher_team_id", "batter_team_id", "count_state", "hand_matchup",
    ]
    for c in cats:
        x[c] = x[c].fillna("__MISSING__").astype(str)
    return x


def a1_features(raw, aux_bundle):
    x_aux = aux_pitch_base_frame(raw).reindex(columns=aux_bundle["columns"])
    probs = aux_bundle["model"].predict_proba(x_aux)
    if probs.shape[1] != len(PITCH_GROUPS):
        fixed = np.zeros((len(x_aux), len(PITCH_GROUPS)), dtype="float64")
        for j, cls in enumerate(aux_bundle["classes"]):
            fixed[:, int(cls)] = probs[:, j]
        probs = fixed
    aux = pd.DataFrame(index=raw.index)
    for i, g in enumerate(PITCH_GROUPS):
        aux[f"a1_pred_{g}"] = probs[:, i].astype("float32")
        aux[f"a1_delta_{g}"] = (probs[:, i] - raw[f"asof_pitcher_{g}_rate"].fillna(1 / 3).to_numpy(dtype="float64")).astype("float32")
    clipped = np.clip(probs, 1e-7, 1)
    aux["a1_entropy"] = (-(clipped * np.log(clipped)).sum(axis=1)).astype("float32")
    aux["a1_max"] = probs.max(axis=1).astype("float32")
    aux["a1_margin"] = (np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]).astype("float32")
    return aux


def a2_features(raw, aux_bundle):
    x_aux = aux_pitch_base_frame(raw).reindex(columns=aux_bundle["columns"])
    pred_scaled = aux_bundle["model"].predict(x_aux)
    if pred_scaled.ndim == 1:
        pred_scaled = pred_scaled.reshape(-1, len(aux_bundle["target_cols"]))
    pred = pred_scaled * np.asarray(aux_bundle["target_std"], dtype="float64") + np.asarray(aux_bundle["target_mean"], dtype="float64")
    aux = pd.DataFrame(index=raw.index)
    for i, col in enumerate(aux_bundle["target_cols"]):
        aux[f"a2_pred_{col}"] = pred[:, i].astype("float32")
    aux["a2_physical_fallback"] = np.zeros(len(raw), dtype="float32")
    speed = aux["a2_pred_rel_speed"].to_numpy(dtype="float64")
    spin = aux["a2_pred_spin_rate"].to_numpy(dtype="float64")
    aux["a2_speed_spin_ratio"] = (speed / np.maximum(spin, 1.0)).astype("float32")
    aux["a2_speed_zone_gap"] = (
        aux["a2_pred_rel_speed"].to_numpy(dtype="float64")
        - aux["a2_pred_zone_speed"].to_numpy(dtype="float64")
    ).astype("float32")
    aux["a2_power_movement"] = (
        aux["a2_pred_rel_speed"].to_numpy(dtype="float64")
        * aux["a2_pred_movement_norm"].to_numpy(dtype="float64")
    ).astype("float32")
    return aux


def predict_student(raw, bundle, extra):
    x = build_row_features(raw)
    x = add_season_features(x, raw, bundle["seasonal_states"])
    x = add_context_from_state(x, raw, bundle["context_state"])
    e = extra.copy()
    e.index = x.index
    x = pd.concat([x, e], axis=1)
    return np.clip(bundle["model"].predict_proba(x.reindex(columns=bundle["columns"]))[:, 1], 1e-5, 1 - 1e-5)


def predict_catboost(raw, bundle, extra=None):
    x = build_row_features(raw)
    x = add_season_features(x, raw, bundle["seasonal_states"])
    x = add_context_from_state(x, raw, bundle["context_state"])
    if extra is not None:
        e = extra.copy()
        e.index = x.index
        x = pd.concat([x, e], axis=1)
    x_pitch = add_pitchmix(x.copy(), raw, bundle["pitchmix_multiplier"])
    predictions = []
    for i, spec in enumerate(bundle["models"]):
        source = x if i == 0 else x_pitch
        raw_pred = spec["model"].predict_proba(source.reindex(columns=spec["columns"]))[:, 1]
        predictions.append(spec["weight"] * logit_shift(raw_pred, spec["logit_shift"]))
    pred = np.sum(predictions, axis=0)
    pred = pred + bundle["residual_alpha"] * residual_correction(raw, bundle["residual_maps"])
    return np.clip(bundle["center"] + bundle["amplitude"] * (pred - bundle["center"]), 1e-5, 1 - 1e-5)


def main():
    test = pd.read_csv(find_path(Path("data/test.csv")), encoding="utf-8-sig")
    champion_bundle = joblib.load(find_path(Path("model/catboost_original_bundle.pkl")))
    a1_bundle = joblib.load(find_path(Path("model/a1_catboost_bundle.pkl")))
    a2_bundle = joblib.load(find_path(Path("model/a2_aux_bundle.pkl")))
    d1_bundle = joblib.load(find_path(Path("model/d1_student_bundle.pkl")))
    rf_meta, rf_trees = load_portable_rf()

    p_rf = predict_portable_rf(build_rf_exp01_features(test), rf_meta, rf_trees)
    p_champion_cb = predict_catboost(test, champion_bundle)
    p_champion = CATBOOST_WEIGHT * p_champion_cb + RF_WEIGHT * p_rf
    extra = pd.concat([a1_features(test, a1_bundle["aux_pitch"]), a2_features(test, a2_bundle)], axis=1)
    p_student = predict_student(test, d1_bundle, extra)
    pred = (1 - D1_BLEND_WEIGHT) * p_champion + D1_BLEND_WEIGHT * p_student

    out = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: pred})
    output_dir = BASE_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    out.to_csv(output_dir / "submission.csv", index=False, encoding="utf-8")
    print(f"saved {output_dir / 'submission.csv'} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
