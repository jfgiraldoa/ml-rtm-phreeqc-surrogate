"""Train DT, RF, XGBoost surrogates of one PHREEQC reaction step and report per-species metrics and timing.

Usage (ml-rtm env): python models/train_eval.py [dataset.csv] [heldout_chain]
Formulation: inputs = totals sent to react (Ca, C, Cl, Mg) + calcite + dolomite of the cell (6 features).
Targets = change over the step of Ca, C, Mg, calcite, dolomite and the absolute pH, standardised; Cl is
conserved (out = in). Absolute outputs are rebuilt as in + change, which keeps the mass balance structure.
Hyperparameters are fixed (not tuned on this data); XGBoost uses subsample = 1.0 because 0.6 breaks transport loops.
Split cells: 80/20 by cell. Split heldout_front: train on all chains but one benchmark-like front chain, test on it.
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "models" / "results").mkdir(parents=True, exist_ok=True)
data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw/dataset_engesgaard1d_v2_32k_seed42.csv"
HOLDOUT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tag = data_path.stem

df = pd.read_csv(data_path)
df["group"] = df["chain"] * 100000 + df["cell"]
FEATURES = ["in_Ca", "in_C", "in_Cl", "in_Mg", "in_calcite", "in_dolomite"]
SPECIES = ["Ca", "C", "Mg", "calcite", "dolomite"]
X = df[FEATURES].values
Y = np.c_[[(df[f"out_{k}"] - df[f"in_{k}"]).values for k in SPECIES] + [df["out_pH"].values]].T   # changes + pH
NAMES = SPECIES + ["pH"]
splits = {"cells": next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, groups=df["group"])),
          "heldout_front": (np.flatnonzero(df.chain != HOLDOUT), np.flatnonzero(df.chain == HOLDOUT))}

models = {
    "DT": lambda: DecisionTreeRegressor(max_depth=20, max_features=1.0, min_samples_leaf=4, min_samples_split=2, random_state=42),
    "RF": lambda: RandomForestRegressor(n_estimators=75, max_depth=20, max_features=1.0, min_samples_leaf=2,
                                        min_samples_split=2, random_state=42, n_jobs=1),
    "XGB": lambda: XGBRegressor(colsample_bytree=1.0, learning_rate=0.05, max_depth=10, n_estimators=500, reg_lambda=0.1,
                                subsample=1.0, random_state=42, n_jobs=1),
}

rows, kept = [], {}
for split, (tr, te) in splits.items():
    sc = StandardScaler().fit(Y[tr])
    print(f"{tag} [{split}]: train {len(tr)}, test {len(te)} rows", flush=True)
    preds = {}
    for name in ["no-reaction baseline"] + list(models):
        if name.startswith("no-reaction"):
            p = np.c_[np.zeros((len(te), 5)), np.full(len(te), Y[tr][:, 5].mean())]
            t_fit = t_pred = 0.0
        else:
            m = models[name]()
            t = time.time(); m.fit(X[tr], sc.transform(Y[tr])); t_fit = time.time() - t
            t = time.time(); p = sc.inverse_transform(m.predict(X[te])); t_pred = time.time() - t
            joblib.dump({"model": m, "y_scaler": sc, "features": FEATURES, "targets": NAMES},
                        ROOT / f"models/results/{tag}_{split}_{name}.joblib")
        preds[name] = p
        r = {"split": split, "model": name, "fit_s": round(t_fit, 1), "pred_ms_per_cell": 1e3 * t_pred / len(te)}
        for i, k in enumerate(NAMES):
            r[f"R2change_{k}" if k != "pH" else "R2_pH"] = r2_score(Y[te][:, i], p[:, i])
        for k in ("Ca", "calcite", "dolomite"):
            i = NAMES.index(k)
            r[f"R2abs_{k}"] = r2_score(df[f"out_{k}"].values[te], df[f"in_{k}"].values[te] + p[:, i])
        rows.append(r)
        print(" ", name, flush=True)
    kept[split] = (Y[te], preds)

res = pd.DataFrame(rows)
res.to_csv(ROOT / f"models/results/{tag}_metrics.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.set_index(["split", "model"]).round(4).T.to_string())

# PHREEQC time per cell, same engine setup (100 cells as in the benchmark and a 2000-cell batch)
sys.path.insert(0, str(ROOT / "data_generation"))
import json
import generate_dataset as g
cfg = json.loads((ROOT / "data_generation/config_engesgaard1d_v2.json").read_text())
for n in (100, 2000):
    c = g.Cells(dict(cfg, n_cells=n)); st = c.state(); rng = np.random.default_rng(0)
    new = {k: st[k] * (1 + 0.01 * rng.random(n)) for k in g.COMP}
    t = time.time(); c.react(new); dt = time.time() - t
    print(f"PHREEQC react: {n} cells {dt:.4f} s -> {1e3 * dt / n:.4f} ms/cell")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for split, (yte, preds) in kept.items():
    fig, ax = plt.subplots(3, len(NAMES), figsize=(2.9 * len(NAMES), 9.3))
    for r_, name in enumerate(models):
        for c_, k in enumerate(NAMES):
            a = ax[r_, c_]; a.scatter(yte[:, c_], preds[name][:, c_], s=2, alpha=0.4)
            lo, hi = yte[:, c_].min(), yte[:, c_].max(); a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            a.set_title(f"{name} {'d' + k if k != 'pH' else 'pH'}  R2={r2_score(yte[:, c_], preds[name][:, c_]):.4f}", fontsize=8)
            if r_ == 2: a.set_xlabel("PHREEQC")
            if c_ == 0: a.set_ylabel("surrogate")
    fig.suptitle(f"{tag} [{split}]  (d = change over the step)", fontsize=10)
    fig.tight_layout(); fig.savefig(ROOT / f"models/results/{tag}_{split}_parity.png", dpi=105); plt.close(fig)
print("saved metrics, models and figures in models/results/")
