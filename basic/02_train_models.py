"""Step 2. Train a random forest to replace one PHREEQC reaction step.

Inputs  (10): Ca, C, Cl, Mg totals sent to PHREEQC (mol/L), calcite and dolomite already in the cell, and log10 of
              the four totals (see common.py for why).
Targets  (6): CHANGE over the step of Ca, C, Mg, calcite, dolomite, and the pH after the step.
              Cl does not react, so it is not predicted (out = in).
Why the change and not the final value: dolomite barely changes in one step, so a model that predicts the final
value looks perfect (R2 ~ 0.99) even if it learns nothing. The change is the part PHREEQC actually computes.

Hyperparameters are fixed (not tuned on this data).
Split: 80 % of the cells for training, 20 % for testing; a cell and all its steps stay on the same side.

Run:  python 02_train_models.py      (about 10 s)
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from common import make_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
(HERE / "results").mkdir(exist_ok=True)
df = pd.read_csv(HERE / "data" / "engesgaard_columns.csv")

CHANGES = ["Ca", "C", "Mg", "calcite", "dolomite"]
X = make_features(*(df[f"in_{k}"].values for k in ["Ca", "C", "Cl", "Mg", "calcite", "dolomite"]))
Y = np.column_stack([(df[f"out_{k}"] - df[f"in_{k}"]).values for k in CHANGES] + [df["out_pH"].values])
TARGETS = [f"d{k}" for k in CHANGES] + ["pH"]

cell_id = df["column"] * 1000 + df["cell"]
train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, groups=cell_id))
scaler = StandardScaler().fit(Y[train])          # targets have very different sizes (1e-7 ... 10), so we standardise them
print(f"train {len(train)} rows, test {len(test)} rows (cells never seen in training)")

model = RandomForestRegressor(n_estimators=75, max_depth=20, max_features=1.0, min_samples_leaf=2, min_samples_split=2,
                              random_state=42, n_jobs=1)
t0 = time.time()
model.fit(X[train], scaler.transform(Y[train]))
fit_s = time.time() - t0
print(f"model 1/1  RandomForest  fit {fit_s:.1f} s", flush=True)

t0 = time.time()
pred = scaler.inverse_transform(model.predict(X[test]))
pred_ms = 1e3 * (time.time() - t0) / len(test)
row = {"fit_s": round(fit_s, 1), "predict_ms_per_cell": round(pred_ms, 5)}
for j, t in enumerate(TARGETS):
    row[f"R2_{t}"] = round(r2_score(Y[test][:, j], pred[:, j]), 4)
results = pd.DataFrame([row], index=["RandomForest"])
results.to_csv(HERE / "results" / "metrics.csv")
joblib.dump({"model": model, "scaler": scaler, "targets": TARGETS}, HERE / "results" / "RandomForest.joblib")
np.savez(HERE / "results" / "test_predictions.npz", y_true=Y[test], targets=TARGETS, RandomForest=pred)
pd.set_option("display.width", 200)
print(results.T.to_string())
