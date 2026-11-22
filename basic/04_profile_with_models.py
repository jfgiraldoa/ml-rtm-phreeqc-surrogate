"""Step 4. Run the Engesgaard benchmark column with the trained forest instead of PHREEQC for the chemistry.

Same transport as step 1 (Courant 1.0, diffusion number 1.339, 40 steps of 533 s). At every step the totals are
moved by the transport and then the chemistry is done either by PHREEQC (reference) or by the forest, which predicts the
change of Ca, C, Mg, calcite, dolomite and the pH. Errors of the model accumulate along the 40 steps, so this is a
harder test than the test set of step 2. The benchmark column is not in the training data. Negative values are clipped to zero (the count is printed).

Run:  python 04_profile_with_models.py     (needs steps 1 and 2)
"""
import importlib.util
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from common import make_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("gen", HERE / "01_generate_data.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

COURANT, DIFFUSION = gen.BENCHMARK              # not in the training columns
TO_BULK = 1000 * gen.POROSITY / (1 - gen.POROSITY)        # mol per cell -> mol/m3 of bulk volume (plot units only)
x = (np.arange(gen.N_CELLS) + 0.5) * 0.005


def profile(s):
    return {"Ca": 1000 * s["Ca"], "Mg": 1000 * s["Mg"], "calcite": TO_BULK * s["calcite"],
            "dolomite": TO_BULK * s["dolomite"], "pH": s["pH"]}


# reference: PHREEQC in every cell
t0 = time.time()
column = gen.Column(gen.N_CELLS)
start = column.state()
for step in range(gen.N_STEPS):
    print(f"run 1/2  step {step + 1}/{gen.N_STEPS}  {time.time() - t0:.1f} s  PHREEQC", flush=True)
    column.react(gen.transport(column.state(), COURANT, DIFFUSION))
curves = {"PHREEQC": profile(column.state())}

# surrogate: same start, chemistry replaced by the forest
bundle = joblib.load(HERE / "results" / "RandomForest.joblib")
s = {key: start[key].copy() for key in start}
clipped = 0
for step in range(gen.N_STEPS):
    print(f"run 2/2  step {step + 1}/{gen.N_STEPS}  {time.time() - t0:.1f} s  RandomForest", flush=True)
    tot = gen.transport(s, COURANT, DIFFUSION)
    feats = make_features(tot["Ca"], tot["C"], tot["Cl"], tot["Mg"], s["calcite"], s["dolomite"])
    d = bundle["scaler"].inverse_transform(bundle["model"].predict(feats))      # dCa, dC, dMg, dcalcite, ddolomite, pH
    new = {"Ca": tot["Ca"] + d[:, 0], "C": tot["C"] + d[:, 1], "Cl": tot["Cl"], "Mg": tot["Mg"] + d[:, 2],
           "calcite": s["calcite"] + d[:, 3], "dolomite": s["dolomite"] + d[:, 4], "pH": d[:, 5]}
    for key in ("Ca", "C", "Mg", "calcite", "dolomite"):
        clipped += int((new[key] < 0).sum())
        new[key] = np.clip(new[key], 0.0, None)
    s = new
curves["RandomForest"] = profile(s)
ref = curves["PHREEQC"]
mx = {key: np.abs(curves["RandomForest"][key] - ref[key]).max() / (ref[key].max() - ref[key].min())
      for key in ("Ca", "calcite", "dolomite", "pH")}
l2 = {key: np.linalg.norm(curves["RandomForest"][key] - ref[key]) / np.linalg.norm(ref[key])
      for key in ("Ca", "calcite", "dolomite")}
print("RandomForest: max |error| / range of the PHREEQC profile  " + "  ".join(f"{a} {b:.3f}" for a, b in mx.items())
      + "   | relative L2  " + "  ".join(f"{a} {b:.3f}" for a, b in l2.items()) + f"   (clipped values: {clipped})")

np.savez(HERE / "results" / "column_profiles.npz", x=x, **{f"{n}_{k}": v for n, c in curves.items() for k, v in c.items()})
fig, ax = plt.subplots(1, 5, figsize=(19, 3.4))
for a, key, title in zip(ax, ("Ca", "Mg", "calcite", "dolomite", "pH"),
                         ("Ca (mol/m3)", "Mg (mol/m3)", "calcite (mol/m3 bulk)", "dolomite (mol/m3 bulk)", "pH")):
    a.plot(x, curves["PHREEQC"][key], "k-", lw=2, label="PHREEQC")
    a.plot(x, curves["RandomForest"][key], "r--", label="RandomForest")
    a.set_title(title + ", t = 40 steps"); a.set_xlabel("x (m)")
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(HERE / "results" / "column_with_models.png", dpi=110)
print("saved results/column_with_models.png")
