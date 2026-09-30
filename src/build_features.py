"""Build per-reference-check features for all 28 batteries.

For every reference discharge, compute how much the battery had been used up
to that point (cumulative Ah throughput, elapsed hours) and the stress it saw
since the previous check (temperature, discharge current). Joins the cleaned
SOH labels, so the 11 removed points are dropped automatically.

Back-to-back reference discharges with no usage in between (the same check-up
measured twice, ~7.5 h apart) are merged into one check: capacity is averaged
and SOH is recomputed relative to each battery's first merged check.

Run from the project root:
    python src/build_features.py
"""
import glob
import os

import numpy as np
import pandas as pd
import scipy.io as sio

RAW = "data/raw/groups"
LABELS = "data/processed/reference_capacity_clean.csv"
OUT = "data/processed/features.csv"


def arr(x):
    """Return x as a 1-D float array (single-sample steps load as scalars)."""
    return np.atleast_1d(np.asarray(x, dtype=float))


T_MIN, T_MAX = -20.0, 80.0  # plausible cell temperatures (deg C); outside = sensor glitch


def valid_temp_mean(temperature):
    """Mean temperature, ignoring physically impossible sensor readings."""
    t = arr(temperature)
    t = t[(t > T_MIN) & (t < T_MAX)]
    return float(np.mean(t)) if len(t) else np.nan


def amp_hours(current, time):
    """Charge moved in a step, in Ah (absolute value, so charge and discharge both count)."""
    i, t = arr(current), arr(time)
    if len(t) < 2:
        return 0.0
    return float(np.trapezoid(np.abs(i), t) / 3600.0)


def battery_features(path):
    name = os.path.basename(path)[:-4]
    group = path.split(os.sep)[3]  # data/raw/groups/<g1..g7>/...
    steps = sio.loadmat(path, simplify_cells=True)["data"]["step"]
    t0 = arr(steps[0]["time"])[0]

    rows = []
    cum_ah = 0.0
    ref_idx = 0
    # accumulators for the interval since the previous reference discharge
    iv_ah, iv_temp, iv_dis_i, iv_steps = 0.0, [], [], 0

    for st in steps:
        comment = st["comment"]

        if comment == "reference discharge":
            t = arr(st["time"])
            rows.append({
                "battery": name,
                "group": group,
                "ref_idx": ref_idx,
                "elapsed_h": (t[0] - t0) / 3600.0,
                "cum_ah": cum_ah,
                "interval_ah": iv_ah,
                "interval_rw_steps": iv_steps,
                "interval_mean_temp": float(np.mean(iv_temp)) if iv_temp else np.nan,
                "interval_max_temp": float(np.max(iv_temp)) if iv_temp else np.nan,
                "interval_mean_dis_current": float(np.mean(iv_dis_i)) if iv_dis_i else np.nan,
                "ref_mean_temp": valid_temp_mean(st["temperature"]),
            })
            ref_idx += 1
            iv_ah, iv_temp, iv_dis_i, iv_steps = 0.0, [], [], 0

        step_ah = amp_hours(st["current"], st["time"])
        cum_ah += step_ah

        if "random walk" in comment:
            iv_ah += step_ah
            step_temp = valid_temp_mean(st["temperature"])
            if not np.isnan(step_temp):
                iv_temp.append(step_temp)
            if st["type"] == "D":
                iv_steps += 1
                iv_dis_i.append(float(np.mean(np.abs(arr(st["current"])))))

    return rows


def merge_repeat_checks(df):
    """Merge consecutive reference checks that had no usage between them."""
    df = df.sort_values(["battery", "ref_idx"]).copy()
    is_first = df.groupby("battery").cumcount() == 0
    new_check = (df["interval_ah"] > 0) | is_first
    df["check_id"] = new_check.astype(int).groupby(df["battery"]).cumsum() - 1

    keep_first = [c for c in df.columns
                  if c not in ("battery", "check_id", "capacity_Ah", "soh", "ref_idx")]
    agg = {c: "first" for c in keep_first}
    agg["capacity_Ah"] = "mean"
    agg["ref_idx"] = "size"
    out = (df.groupby(["battery", "check_id"], as_index=False)
             .agg(agg)
             .rename(columns={"ref_idx": "n_repeats"}))

    first_cap = out.groupby("battery")["capacity_Ah"].transform("first")
    out["soh"] = out["capacity_Ah"] / first_cap * 100.0
    return out


def main():
    files = sorted(glob.glob(os.path.join(RAW, "g*", "*", "data", "Matlab", "*.mat")))
    rows = []
    for f in files:
        print("reading", os.path.basename(f), flush=True)
        rows.extend(battery_features(f))

    feats = pd.DataFrame(rows)
    labels = pd.read_csv(LABELS)[["battery", "ref_idx", "capacity_Ah", "soh"]]
    df = feats.merge(labels, on=["battery", "ref_idx"], how="inner")
    n_before = len(df)
    df = merge_repeat_checks(df)
    df.to_csv(OUT, index=False)

    print(f"merged {n_before} measurements into {len(df)} checks "
          f"for {df.battery.nunique()} batteries -> {OUT}")
    summary = pd.DataFrame({
        "checks": df.groupby("group").size(),
        "missing_temp": df["interval_mean_temp"].isna().groupby(df["group"]).sum(),
    })
    print(summary)
    print("temperature range after filtering: "
          f"{df['interval_mean_temp'].min():.1f} to {df['interval_mean_temp'].max():.1f} C (interval), "
          f"{df['ref_mean_temp'].min():.1f} to {df['ref_mean_temp'].max():.1f} C (reference)")


if __name__ == "__main__":
    main()
