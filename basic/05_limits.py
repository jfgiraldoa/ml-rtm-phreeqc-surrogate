"""Step 5. Where the trained forest stops working: change the conditions it was trained for.

The forest only learned the chemistry of ONE system: this inlet water, this initial water, this time step.
Here we run the benchmark column again with PHREEQC and with the forest, changing one condition at a time, and print
the error of the profile after 40 steps (max |error| / range of the PHREEQC profile).

  - same boundary                  -> control, the error of step 4
  - other flow speed / dispersion  -> the model only sees the state of a cell, so it should still work
  - boundary 0.5x and 3x           -> new mixtures of Ca, C, Mg, Cl that the model never saw
  - time step 2x                   -> dt is not an input of the model, but dolomite is kinetic

This is the point of the exercise: a surrogate trained on one problem does not transfer by itself. To use it on other
conditions it has to be trained over many of them (and dt, boundary values, ... added as inputs).

Run:  python 05_limits.py     (needs steps 1 and 2, about 30 s)
"""
import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
from common import make_features

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("gen", HERE / "01_generate_data.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
bundle = joblib.load(HERE / "results" / "RandomForest.joblib")

BASE = {"Ca": 1e-10, "C": 1e-10, "Cl": 0.002, "Mg": 0.001}         # inlet water used for training
CASES = [  # name, inlet water, time step (s), Courant, diffusion number
    ("control: same boundary, benchmark column", BASE, 533.0, 1.0, 1.339),
    ("other flow speed and dispersion (not in training)", BASE, 533.0, 0.6, 0.3),
    ("boundary 0.5x (Mg 5e-4, Cl 1e-3)", {**BASE, "Cl": 0.001, "Mg": 0.0005}, 533.0, 1.0, 1.339),
    ("boundary 3x (Mg 3e-3, Cl 6e-3)", {**BASE, "Cl": 0.006, "Mg": 0.003}, 533.0, 1.0, 1.339),
    ("time step 2x (1066 s)", BASE, 1066.0, 1.0, 1.339),
]


def run(inlet, dt, courant, diffusion):
    gen.INLET, gen.DT = dict(inlet), dt                  # the functions of step 1 read these two module variables
    column = gen.Column(gen.N_CELLS)
    start = column.state()
    for _ in range(gen.N_STEPS):                         # reference: PHREEQC
        column.react(gen.transport(column.state(), courant, diffusion))
    ref = column.state()
    s = start
    for _ in range(gen.N_STEPS):                         # forest instead of PHREEQC (same loop as step 4)
        tot = gen.transport(s, courant, diffusion)
        feats = make_features(tot["Ca"], tot["C"], tot["Cl"], tot["Mg"], s["calcite"], s["dolomite"])
        d = bundle["scaler"].inverse_transform(bundle["model"].predict(feats))
        s = {"Ca": tot["Ca"] + d[:, 0], "C": tot["C"] + d[:, 1], "Cl": tot["Cl"], "Mg": tot["Mg"] + d[:, 2],
             "calcite": s["calcite"] + d[:, 3], "dolomite": s["dolomite"] + d[:, 4], "pH": d[:, 5]}
        for k in ("Ca", "C", "Mg", "calcite", "dolomite"):
            s[k] = np.clip(s[k], 0.0, None)
    return [np.abs(s[k] - ref[k]).max() / (ref[k].max() - ref[k].min()) for k in ("Ca", "calcite", "dolomite", "pH")]


print(f"{'case':52s} {'Ca':>7s} {'calcite':>8s} {'dolomite':>9s} {'pH':>7s}   max |error| / range of the PHREEQC profile")
for i, (name, inlet, dt, courant, diffusion) in enumerate(CASES, start=1):
    print(f"run {i}/{len(CASES)}", flush=True, file=sys.stderr)
    e = run(inlet, dt, courant, diffusion)
    print(f"{name:52s} " + " ".join(f"{v:7.1%}" for v in e), flush=True)
