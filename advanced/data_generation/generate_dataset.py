"""Dataset generator that reproduces the chemistry call of PhreeqcRM-based reactive-transport engines (case eq2d or engesgaard1d).

Cells start as the engine's initial pore water (SOLUTION 10 + calcite) and evolve. At each step every cell
receives new totals of Ca, C, Cl, Mg, built by mixing its own totals with a random other cell and with the
inlet (a stand-in for transport); PhreeqcRM.SetConcentrations + RunCells then equilibrate, exactly as
`EquilibriumCalciteDolomite2DPhreeqcRM.react`. H, O and minerals stay in the cell. One row per cell per step.

Usage (ml-rtm env): python generate_eq2d.py [config.json]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from phreeqcrm import PhreeqcRM

HERE = Path(__file__).parent
COMP = ("Ca", "C", "Cl", "Mg")


RATES_ENGESGAARD = """
RATES
    Dolomite
    -start
    10 logK25 = -3.19
    20 ny = 0.5
    30 mech_a = (10^logK25) * ACT("H+")^ny
    40 logK25 = -7.53
    50 mech_b = (10^logK25)
    60 rate = mech_a + mech_b
    70 teta = 1
    80 eta = 1
    90 Area = 0.001
    100 rate = Area * rate * (1 - SR("Dolomite")^teta)^eta
    110 delta_moles = rate * TIME
    120 SAVE delta_moles
    -end
"""


def phreeqc_input(cfg):
    i, o = cfg["inlet"], cfg["initial"]
    if cfg["case"] == "engesgaard1d":
        return f"""
SOLUTION 0 Engesgaard inflow
    temp 25
    pH 7 charge
    pe 4
    units mol/kgw
    Ca {i['Ca']}
    C {i['C']}
    Cl {i['Cl']}
    Mg {i['Mg']}

SOLUTION 10 Engesgaard initial pore water
    temp 25
    pH 7 charge
    pe 4
    units mol/kgw
    Ca {o['Ca']}
    C {o['C']}
    Cl {o['Cl']}
    Mg {o['Mg']}
{RATES_ENGESGAARD}
EQUILIBRIUM_PHASES 1
    Calcite 0 {o['calcite']:.7e}

KINETICS 1
    Dolomite
        -m 0
        -m0 0
END
"""
    return f"""
SOLUTION 0 2D calcite-dolomite inlet
    temp 25
    pH 7 charge
    pe 4
    units mol/kgw
    Ca {i['Ca']}
    C {i['C']}
    Cl {i['Cl']}
    Mg {i['Mg']}

SOLUTION 10 2D calcite-dolomite initial pore water
    temp 25
    pH 7 charge
    pe 4
    units mol/kgw
    Ca {o['Ca']}
    C {o['C']}
    Cl {o['Cl']}
    Mg {o['Mg']}

EQUILIBRIUM_PHASES 1
    Calcite 0 {o['calcite']:.8e}
    Dolomite 0 {o['dolomite']:.8e}
END
"""


SELECTED_ENGESGAARD = """
SELECTED_OUTPUT 1
    -reset false
    -high_precision true
USER_PUNCH 1
    -headings Calcite Dolomite pH SI_Calcite SI_Dolomite water_kg
    10 PUNCH EQUI("Calcite"), KIN("Dolomite"), -LA("H+"), SI("Calcite"), SI("Dolomite"), TOT("water")
END
"""

SELECTED = """
SELECTED_OUTPUT 1
    -reset false
    -high_precision true
USER_PUNCH 1
    -headings Calcite Dolomite pH SI_Calcite SI_Dolomite water_kg
    10 PUNCH EQUI("Calcite"), EQUI("Dolomite"), -LA("H+"), SI("Calcite"), SI("Dolomite"), TOT("water")
END
"""


class Cells:
    def __init__(self, cfg):
        n, por = cfg["n_cells"], cfg["porosity"]
        self.n = n
        self.dt = cfg["time_step"]
        self.t = 0.0
        rm = self.rm = PhreeqcRM(n, 1)
        rm.SetErrorHandlerMode(0)
        rm.SetScreenOn(False)
        assert rm.LoadDatabase(str((HERE / cfg["database"]).resolve())) == 0
        rm.SetUnitsSolution(2)
        rm.SetUnitsPPassemblage(1)
        rm.SetPorosity(np.full(n, por))
        rm.SetSaturationUser(np.ones(n))
        rm.SetRepresentativeVolume(np.full(n, 1.0 / por))
        rm.UseSolutionDensityVolume(False)
        kinetic = cfg["case"] == "engesgaard1d"
        if kinetic:
            rm.SetUnitsKinetics(1)
        assert rm.RunString(True, True, True, phreeqc_input(cfg)) == 0
        assert rm.RunString(True, False, False, SELECTED_ENGESGAARD if kinetic else SELECTED) == 0
        rm.SetSelectedOutputOn(True)
        assert rm.FindComponents() > 0
        self.comps = [str(c) for c in rm.GetComponents()]
        ic = np.full(7 * n, -1, dtype=np.int32)
        ic[0:n] = 10
        ic[n:2 * n] = 1
        if kinetic:
            ic[6 * n:7 * n] = 1
        assert rm.InitialPhreeqc2Module(ic) == 0
        rm.SetTimeStep(0.0); rm.SetTime(0.0)
        rm.RunCells()

    def matrix(self):
        return np.asarray(self.rm.GetConcentrations()).reshape(len(self.comps), self.n).copy()

    def state(self):
        m = self.matrix()
        self.rm.SetCurrentSelectedOutputUserNumber(1)
        heads = [str(h) for h in self.rm.GetSelectedOutputHeadings()]
        sel = dict(zip(heads, np.asarray(self.rm.GetSelectedOutput()).reshape(len(heads), self.n)))
        s = {c: m[self.comps.index(c)] for c in self.comps}      # all components, mol/L (incl. H, O)
        s.update(calcite=sel["Calcite"], dolomite=sel["Dolomite"], pH=sel["pH"],
                 SI_calcite=sel["SI_Calcite"], SI_dolomite=sel["SI_Dolomite"], water_kg=sel["water_kg"])
        return s

    def react(self, totals):
        m = self.matrix()
        for c in COMP:
            m[self.comps.index(c)] = totals[c]
        assert self.rm.SetConcentrations(m.reshape(-1)) == 0
        self.rm.SetTimeStep(self.dt)
        self.t += self.dt
        self.rm.SetTime(self.t)
        assert self.rm.RunCells() == 0


def transport_step(totals, cr, dn, inlet):
    """1D upwind advection (Courant cr), then implicit diffusion (dn = D dt / dx^2); cell 0 sees the inlet."""
    from scipy.linalg import solve_banded
    new = {}
    for c in COMP:
        v = totals[c]
        n = len(v)
        moved = v - cr * (v - np.r_[inlet[c], v[:-1]])
        band = np.zeros((3, n))
        band[0, 1:] = -dn
        band[1, :] = 1 + 2 * dn
        band[2, :-1] = -dn
        band[1, -1] = 1 + dn
        rhs = moved.copy()
        rhs[0] += dn * inlet[c]
        new[c] = solve_banded((1, 1), band, rhs)
    return new


def run_chain(cfg, spec, rng, chain, n_chains, t0):
    """One chain of cells evolving for spec['n_steps'] steps; spec['type'] is 'mix' or 'front'."""
    spec = dict(spec)
    spec.setdefault("n_cells", cfg["n_cells"])
    ccfg = dict(cfg, n_cells=spec["n_cells"])
    cells = Cells(ccfg)
    rows = []
    for step in range(spec["n_steps"]):
        print(f"chain {chain + 1}/{n_chains} ({spec['type']}) step {step + 1}/{spec['n_steps']}  "
              f"t={time.time() - t0:.1f}s", flush=True)
        before = cells.state()
        if spec["type"] == "front":
            new = transport_step(before, spec["courant"], spec["diffusion_number"], cfg["inlet"])
        else:
            other = rng.permutation(cells.n)
            a = rng.random(cells.n) * cfg["neighbour_fraction_max"]
            f = rng.random(cells.n) * spec["inlet_fraction_max"]
            new = {}
            for c in COMP:
                t = (1 - a) * before[c] + a * before[c][other]        # exchange with another cell
                new[c] = (1 - f) * t + f * cfg["inlet"][c]            # inlet mixing
        cells.react(new)
        after = cells.state()
        row = {f"in_{c}": new[c] for c in COMP}
        row.update(in_calcite=before["calcite"], in_dolomite=before["dolomite"], in_pH=before["pH"],
                   in_H=before["H"], in_O=before["O"], dt=np.full(cells.n, cfg["time_step"]),
                   step=np.full(cells.n, step), chain=np.full(cells.n, chain), cell=np.arange(cells.n),
                   chain_type=np.full(cells.n, spec["type"]))
        row.update({f"out_{c}": after[c] for c in COMP})
        row.update(out_calcite=after["calcite"], out_dolomite=after["dolomite"], out_pH=after["pH"],
                   out_SI_calcite=after["SI_calcite"], out_SI_dolomite=after["SI_dolomite"])
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def main(path):
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault("case", "eq2d")
    if "chains" not in cfg:                                  # legacy layout: one mixing chain per fraction
        cfg["neighbour_fraction_max"] = cfg["mixing"]["neighbour_fraction_max"]
        cfg["chains"] = [{"type": "mix", "inlet_fraction_max": f, "n_steps": cfg["n_steps"]}
                         for f in cfg["mixing"]["inlet_fraction_max_list"]]
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    chains = [run_chain(cfg, sp, rng, k, len(cfg["chains"]), t0) for k, sp in enumerate(cfg["chains"])]
    df = pd.concat(chains, ignore_index=True)

    # mass balance (water = 1 L per cell; minerals in mol/reactor both before and after)
    for name, a_, b_ in (
        ("Ca", df.in_Ca + df.in_calcite + df.in_dolomite, df.out_Ca + df.out_calcite + df.out_dolomite),
        ("Mg", df.in_Mg + df.in_dolomite, df.out_Mg + df.out_dolomite),
        ("C", df.in_C + df.in_calcite + 2 * df.in_dolomite, df.out_C + df.out_calcite + 2 * df.out_dolomite),
    ):
        e = (a_ - b_).abs() / a_.clip(lower=1e-30)
        print(f"balance {name}: max rel err {e.max():.2e}, frac > 1e-2: {(e > 1e-2).mean():.4f}")
    print("rows", len(df), "| pH out", df.out_pH.min().round(2), df.out_pH.max().round(2),
          "| calcite>0", (df.out_calcite > 1e-12).mean().round(3), "dolomite>0", (df.out_dolomite > 1e-12).mean().round(3),
          "| max dolomite", f"{df.out_dolomite.max():.2e}")
    out = HERE.parent / "data" / "raw" / f"dataset_{cfg['name']}_{len(df) // 1000}k_seed{cfg['seed']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "config_eq2d.json")
