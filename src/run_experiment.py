"""Run model arms across every fold, label budget and seed, and save predictions.

Run from the project root, for example:
    python src/run_experiment.py --arms ml_mlp ml_gbm
    python src/run_experiment.py --arms ml_mlp --quick     # 1 fold, 1 seed: smoke test

Outputs one file per arm: results/pred_<arm>.csv
(fold, budget, seed, battery, check_id, soh_true, soh_pred)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from common import load_data, split, run_errors
from models import ARMS

BUDGETS = [1.0, 0.5, 0.25, 0.10]
OUT_DIR = "results"


def run_arm(arm, df, folds, masks, fold_list, seed_list):
    fit = ARMS[arm]
    rows = []
    start = time.time()
    for fold in fold_list:
        for budget in BUDGETS:
            for seed in seed_list:
                train_lab, train_all, test = split(df, folds, masks, fold, budget, seed)
                predict = fit(train_lab, train_all, seed)
                out = test[["battery", "check_id", "soh_frac"]].rename(columns={"soh_frac": "soh_true"})
                out = out.assign(fold=fold, budget=budget, seed=seed, soh_pred=predict(test))
                rows.append(out)
                rmse, _ = run_errors(out)
                print(f"  {arm} fold {fold} budget {budget:>4} seed {seed}: "
                      f"{len(train_lab):>3} labels, test RMSE {rmse:5.2f} pts "
                      f"({time.time() - start:5.0f}s)", flush=True)
    return pd.concat(rows, ignore_index=True)


def summarise(pred, arm):
    runs = (pred.groupby(["budget", "fold", "seed"])
                .apply(lambda g: pd.Series(run_errors(g), index=["rmse", "mae"]))
                .reset_index())
    table = runs.groupby("budget")[["rmse", "mae"]].agg(["mean", "std"]).round(2)
    print(f"\n{arm}: test error in SOH percentage points (mean and std across runs)")
    print(table.sort_index(ascending=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, choices=list(ARMS))
    ap.add_argument("--quick", action="store_true", help="fold 0 and seed 0 only")
    args = ap.parse_args()

    df, folds, masks = load_data()
    fold_list = [0] if args.quick else sorted(folds["fold"].unique())
    seed_list = [0] if args.quick else sorted(masks["seed"].unique())
    os.makedirs(OUT_DIR, exist_ok=True)

    for arm in args.arms:
        pred = run_arm(arm, df, folds, masks, fold_list, seed_list)
        suffix = "_quick" if args.quick else ""
        path = os.path.join(OUT_DIR, f"pred_{arm}{suffix}.csv")
        pred.to_csv(path, index=False)
        summarise(pred, arm)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
