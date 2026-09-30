"""Experiment design: which batteries are tested, and which labels training may use.

1. Folds: 4-fold cross-validation by battery. Each group has 4 batteries, so
   fold k holds out the k-th battery of every group (7 test batteries per fold).
   Across the 4 folds, every battery is tested exactly once, always unseen.

2. Label budgets: within the training batteries of each fold, keep only a
   fraction of the capacity labels (100%, 50%, 25%, 10%). The first check of
   each battery is always kept, because SOH is defined relative to it.
   Kept labels are spread across the battery's life: its later checks are cut
   into equal bins and one random check is taken from each bin. This mimics
   "fewer reference tests over the same life", not "only early labels".
   Each budget is drawn with several random seeds for the statistical test.

Run from the project root:
    python src/make_splits.py
"""
import numpy as np
import pandas as pd

FEATURES = "data/processed/features.csv"
FOLDS_OUT = "data/processed/folds.csv"
MASKS_OUT = "data/processed/label_masks.csv"

BUDGETS = [1.0, 0.5, 0.25, 0.10]
SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 4


def battery_number(name):
    return int(name.replace("RW", ""))


def make_folds(df):
    batt = df[["battery", "group"]].drop_duplicates().copy()
    batt["num"] = batt["battery"].map(battery_number)
    batt = batt.sort_values(["group", "num"])
    batt["fold"] = batt.groupby("group").cumcount() % N_FOLDS
    return batt[["battery", "group", "fold"]].reset_index(drop=True)


def pick_labels(check_ids, budget, rng):
    """Always keep the first check; spread the rest evenly across life."""
    first, later = check_ids[0], check_ids[1:]
    if budget >= 1.0 or len(later) == 0:
        return set(check_ids)
    k = max(1, int(round(budget * len(later))))
    bins = np.array_split(later, k)
    chosen = {int(rng.choice(b)) for b in bins if len(b) > 0}
    return {first} | chosen


def make_masks(df, folds):
    fold_of = dict(zip(folds["battery"], folds["fold"]))
    rows = []
    for fold in range(N_FOLDS):
        train_batts = [b for b, f in fold_of.items() if f != fold]
        for budget in BUDGETS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed * 1000 + fold * 10 + int(budget * 100))
                for b in train_batts:
                    ids = df.loc[df["battery"] == b, "check_id"].sort_values().to_numpy()
                    keep = pick_labels(ids, budget, rng)
                    for cid in ids:
                        rows.append((fold, budget, seed, b, int(cid), int(cid) in keep))
    return pd.DataFrame(rows, columns=["fold", "budget", "seed", "battery", "check_id", "labelled"])


def main():
    df = pd.read_csv(FEATURES)
    folds = make_folds(df)
    folds.to_csv(FOLDS_OUT, index=False)

    masks = make_masks(df, folds)
    masks.to_csv(MASKS_OUT, index=False)

    print("Test batteries per fold:")
    for f, g in folds.groupby("fold"):
        print(f"  fold {f}: {', '.join(g['battery'])}")

    per_batt = (masks[masks["labelled"]]
                .groupby(["budget", "fold", "seed", "battery"]).size()
                .groupby("budget").agg(["mean", "min", "max"]).round(1))
    total = (masks[masks["labelled"]]
             .groupby(["budget", "fold", "seed"]).size()
             .groupby("budget").mean().round(0).rename("total_train_labels"))
    print("\nLabels per training battery, and total training labels per run:")
    print(per_batt.join(total).sort_index(ascending=False))
    print(f"\nsaved {FOLDS_OUT} and {MASKS_OUT} ({len(masks)} rows)")


if __name__ == "__main__":
    main()
