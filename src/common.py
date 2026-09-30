"""Shared data preparation and evaluation, used identically by every model arm.

Each check becomes one row with five inputs describing the battery's history
up to that check, and one target (SOH, as a fraction of the first check).

Inputs:
    cum_ah        total charge moved so far (kAh)          -> how much it was used
    elapsed_kh    time since the start of the test (1000 h) -> calendar ageing
    avg_temp      Ah-weighted average temperature so far    -> thermal stress
    avg_current   Ah-weighted average discharge current     -> load stress
    ambient_temp  average temperature during reference tests -> chamber (room / 40 C)
"""
import numpy as np
import pandas as pd

FEATURES = "data/processed/features.csv"
FOLDS = "data/processed/folds.csv"
MASKS = "data/processed/label_masks.csv"

FEATURE_COLS = ["cum_ah", "elapsed_kh", "avg_temp", "avg_current", "ambient_temp"]
TARGET_COL = "soh_frac"


def _weighted_running_mean(values, weights):
    """Running weighted mean that skips missing values (they get zero weight)."""
    missing = np.isnan(values)
    weights = np.where(missing, 0.0, weights)
    values = np.where(missing, 0.0, values)
    num = np.cumsum(values * weights)
    den = np.cumsum(weights)
    out = np.full(len(values), np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def load_data():
    df = pd.read_csv(FEATURES).sort_values(["battery", "check_id"]).reset_index(drop=True)

    temp = df["interval_mean_temp"].fillna(df["ref_mean_temp"]).to_numpy()
    current = df["interval_mean_dis_current"].fillna(0.0).to_numpy()
    w = df["interval_ah"].fillna(0.0).to_numpy()

    parts = []
    for _, idx in df.groupby("battery").indices.items():
        idx = np.sort(idx)
        t = _weighted_running_mean(temp[idx], w[idx])
        t = np.where(np.isnan(t), temp[idx], t)  # before any usage: use the check's own temperature
        c = _weighted_running_mean(current[idx], w[idx])
        c = np.nan_to_num(c, nan=0.0)
        amb = pd.Series(df["ref_mean_temp"].to_numpy()[idx]).expanding().mean().to_numpy()
        parts.append(pd.DataFrame({"avg_temp": t, "avg_current": c, "ambient_temp": amb}, index=idx))
    df = df.join(pd.concat(parts))
    # guard: if a check had no valid temperature at all, carry the battery's neighbouring value
    cols = ["avg_temp", "ambient_temp"]
    df[cols] = df.groupby("battery")[cols].transform(lambda s: s.ffill().bfill())
    # guard: a battery with no valid reference-test temperature at all (RW2) gets its
    # chamber temperature estimated from its own usage temperature minus the typical
    # usage-minus-chamber offset of the other batteries in its group
    gap = df["avg_temp"] - df["ambient_temp"]
    offset = gap.groupby(df["group"]).transform("median")
    df["ambient_temp"] = df["ambient_temp"].fillna(df["avg_temp"] - offset)

    df["cum_ah"] = df["cum_ah"] / 1000.0
    df["elapsed_kh"] = df["elapsed_h"] / 1000.0
    df[TARGET_COL] = df["soh"] / 100.0

    folds = pd.read_csv(FOLDS)
    masks = pd.read_csv(MASKS)
    return df, folds, masks


def split(df, folds, masks, fold, budget, seed):
    """Return (labelled training rows, all training rows, test rows) for one run."""
    test_batts = set(folds.loc[folds["fold"] == fold, "battery"])
    train_all = df[~df["battery"].isin(test_batts)]
    test = df[df["battery"].isin(test_batts)]

    m = masks[(masks["fold"] == fold) & np.isclose(masks["budget"], budget)
              & (masks["seed"] == seed) & masks["labelled"]]
    train_lab = train_all.merge(m[["battery", "check_id"]], on=["battery", "check_id"])
    return train_lab, train_all, test


def run_errors(pred):
    """RMSE and MAE in SOH percentage points, on test checks after the first."""
    p = pred[pred["check_id"] > 0]
    err = (p["soh_pred"] - p["soh_true"]) * 100.0
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))
