"""Hybrid test: PHREEQC in the 'active' cells, a trained model (or 'no reaction') in the rest, on the benchmark column.
Active = transported Mg above a threshold, or dolomite present, or calcite partly dissolved (state of the cell only).
PHREEQC runs in all cells here and its result is used only for the active ones, so this measures ACCURACY; the speed
is estimated from the fraction of active cells. Needs basic/ steps 1 and 2 (random forest). Run from anywhere: python hybrid_test.py
"""
import importlib.util, joblib, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "basic"))
from common import make_features
spec=importlib.util.spec_from_file_location("gen",ROOT/"basic/01_generate_data.py"); g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
CR,DN=g.BENCHMARK; N=100; CAL0=1.220005e-4
col0=g.Column(N); s0=col0.state()
for k in range(40): col0.react(g.transport(col0.state(),CR,DN))
ref=col0.state()
B={"RandomForest":joblib.load(ROOT/"basic/results/RandomForest.joblib")}
def run(name,thr,use_mineral_rule=True):
    b=B[name]; col=g.Column(N); s=col.state(); frac=[]
    for k in range(40):
        tot=g.transport(s,CR,DN)
        X=make_features(tot["Ca"],tot["C"],tot["Cl"],tot["Mg"],s["calcite"],s["dolomite"])
        d=b["scaler"].inverse_transform(b["model"].predict(X))
        if name=="no-model": d=np.zeros_like(d); d[:,5]=s["pH"]
        m={"Ca":np.clip(tot["Ca"]+d[:,0],0,None),"C":np.clip(tot["C"]+d[:,1],0,None),"Cl":tot["Cl"],"Mg":np.clip(tot["Mg"]+d[:,2],0,None),
           "calcite":np.clip(s["calcite"]+d[:,3],0,None),"dolomite":np.clip(s["dolomite"]+d[:,4],0,None),"pH":d[:,5]}
        active=(tot["Mg"]>thr)|(s["dolomite"]>1e-10)
        if use_mineral_rule: active|=(s["calcite"]>1e-9)&(s["calcite"]<0.999*CAL0)
        col.react(tot); p=col.state()          # PHREEQC runs all cells here; only the 'active' ones are taken from it
        s={q:np.where(active,p[q],m[q]) for q in ("Ca","C","Cl","Mg","calcite","dolomite","pH")}
        frac.append(active.mean())
    return s,np.mean(frac)
print(f"{'inactive cells':14s} {'Mg threshold':>12s} {'PHREEQC cells':>13s} | max|err|/range:  Ca calcite dolomite pH")
for name in ("no-model","RandomForest"):
    for thr in (1e-4,1e-6,1e-8):
        if name=="no-model": B[name]=B["RandomForest"]
        s_,f=run(name,thr)
        e=[np.abs(s_[k]-ref[k]).max()/(ref[k].max()-ref[k].min()) for k in ("Ca","calcite","dolomite","pH")]
        print(f"{name:14s} {thr:12.0e} {f:12.0%} |   "+"  ".join(f"{v:6.3f}" for v in e),flush=True)
