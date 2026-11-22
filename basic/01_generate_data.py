"""Step 1. Simulate small 1D reactive columns with PHREEQC and record every cell before and after each step.

Case: Engesgaard calcite-dolomite benchmark (standard PhreeqcRM set-up for a reactive-transport code):
calcite in equilibrium, dolomite kinetic, dt = 533 s. A column has 100 cells and runs 40 steps.
Each step has two parts, like in a reactive-transport code:
  (a) transport: move the totals Ca, C, Cl, Mg along the column (upwind advection + implicit diffusion, inlet water at cell 0)
  (b) chemistry: PHREEQC equilibrates every cell with the new totals; this is what the surrogate will learn.
We run 11 columns with different flow speed (Courant number) and dispersion (diffusion number) so the cells see many
chemical states. The Engesgaard benchmark itself (Courant 1.0, diffusion number 1.339) is NOT in the training columns:
step 4 uses it as a test the models have never seen.
Output: data/engesgaard_columns.csv, one row per cell per step (44 000 rows).

Run:  python 01_generate_data.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from phreeqcrm import PhreeqcRM
from scipy.linalg import solve_banded

HERE = Path(__file__).parent
POROSITY, DT, N_CELLS, N_STEPS = 0.32, 533.0, 100, 40
COMPONENTS = ("Ca", "C", "Cl", "Mg")
INLET = {"Ca": 1e-10, "C": 1e-10, "Cl": 0.002, "Mg": 0.001}          # mol/L, water injected at cell 0
# Courant number v*dt/dx and diffusion number D*dt/dx^2 (dx = 0.005 m, v = 9.38e-6 m/s, D = 6.28e-8 m2/s in the benchmark)
# 4 flow speeds x 3 dispersions = 12 combinations, minus the benchmark (1.0, 1.339), which is kept for testing: 11 columns
BENCHMARK = (1.0, 1.339)
COLUMNS = [(cr, dn) for cr in (0.25, 0.5, 0.75, 1.0) for dn in (0.1, 0.5, 1.339) if (cr, dn) != BENCHMARK]

PHREEQC_INPUT = """
SOLUTION 0 inflow
    temp 25
    pH 7 charge
    pe 4
    units mol/kgw
    Ca 1e-10
    C 1e-10
    Cl 0.002
    Mg 0.001

SOLUTION 10 initial pore water
    temp 25
    pH 7 charge
    pe 4
    units mol/kgw
    Ca 1.23e-4
    C 1.23e-4
    Cl 1e-12
    Mg 1e-12

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

EQUILIBRIUM_PHASES 1
    Calcite 0 1.220005e-4

KINETICS 1
    Dolomite
        -m 0
        -m0 0
END
"""

SELECTED_OUTPUT = """
SELECTED_OUTPUT 1
    -reset false
    -high_precision true
USER_PUNCH 1
    -headings Calcite Dolomite pH
    10 PUNCH EQUI("Calcite"), KIN("Dolomite"), -LA("H+")
END
"""


class Column:
    """N PHREEQC reaction cells, set up as inside a reactive-transport code."""

    def __init__(self, n):
        self.n, self.time = n, 0.0
        rm = self.rm = PhreeqcRM(n, 1)
        rm.SetErrorHandlerMode(0)
        rm.SetScreenOn(False)
        assert rm.LoadDatabase(str(HERE / "database" / "phreeqc.dat")) == 0
        rm.SetUnitsSolution(2)                  # mol/L
        rm.SetUnitsPPassemblage(1)
        rm.SetUnitsKinetics(1)
        rm.SetPorosity(np.full(n, POROSITY))
        rm.SetSaturationUser(np.ones(n))
        rm.SetRepresentativeVolume(np.full(n, 1.0 / POROSITY))
        rm.UseSolutionDensityVolume(False)
        assert rm.RunString(True, True, True, PHREEQC_INPUT) == 0
        assert rm.RunString(True, False, False, SELECTED_OUTPUT) == 0
        rm.SetSelectedOutputOn(True)
        assert rm.FindComponents() > 0
        self.names = [str(c) for c in rm.GetComponents()]
        init = np.full(7 * n, -1, dtype=np.int32)
        init[0:n] = 10            # every cell starts as SOLUTION 10
        init[n:2 * n] = 1         # with EQUILIBRIUM_PHASES 1 (calcite)
        init[6 * n:7 * n] = 1     # and KINETICS 1 (dolomite)
        assert rm.InitialPhreeqc2Module(init) == 0
        rm.SetTimeStep(0.0)
        rm.SetTime(0.0)
        rm.RunCells()

    def state(self):
        """Totals (mol/L), calcite, dolomite (mol per cell) and pH of every cell."""
        conc = np.asarray(self.rm.GetConcentrations()).reshape(len(self.names), self.n)
        self.rm.SetCurrentSelectedOutputUserNumber(1)
        sel = np.asarray(self.rm.GetSelectedOutput()).reshape(3, self.n)
        s = {c: conc[self.names.index(c)].copy() for c in COMPONENTS}
        s.update(calcite=sel[0].copy(), dolomite=sel[1].copy(), pH=sel[2].copy())
        return s

    def react(self, totals):
        """Replace the totals Ca, C, Cl, Mg and let PHREEQC react for one step (H, O, minerals stay in the cell)."""
        conc = np.asarray(self.rm.GetConcentrations()).reshape(len(self.names), self.n).copy()
        for c in COMPONENTS:
            conc[self.names.index(c)] = totals[c]
        assert self.rm.SetConcentrations(conc.reshape(-1)) == 0
        self.time += DT
        self.rm.SetTimeStep(DT)
        self.rm.SetTime(self.time)
        assert self.rm.RunCells() == 0


def transport(state, courant, diffusion):
    """One transport step: upwind advection, then implicit diffusion; inlet water enters at cell 0."""
    new = {}
    for c in COMPONENTS:
        v = state[c]
        n = len(v)
        moved = v - courant * (v - np.r_[INLET[c], v[:-1]])           # upwind advection
        band = np.zeros((3, n))                                        # implicit diffusion: (I - dn * Laplacian) c = moved
        band[0, 1:] = -diffusion
        band[1, :] = 1 + 2 * diffusion
        band[2, :-1] = -diffusion
        band[1, -1] = 1 + diffusion                                    # zero gradient at the outlet
        rhs = moved.copy()
        rhs[0] += diffusion * INLET[c]                                 # inlet value enters through the left neighbour
        new[c] = solve_banded((1, 1), band, rhs)
    return new


def main():
    t0 = time.time()
    rows = []
    for col, (courant, diffusion) in enumerate(COLUMNS):
        column = Column(N_CELLS)
        for step in range(N_STEPS):
            print(f"run {col + 1}/{len(COLUMNS)}  step {step + 1}/{N_STEPS}  {time.time() - t0:.1f} s", flush=True)
            before = column.state()
            totals = transport(before, courant, diffusion)
            column.react(totals)
            after = column.state()
            row = {"column": col, "step": step, "cell": np.arange(N_CELLS)}
            row.update({f"in_{c}": totals[c] for c in COMPONENTS})
            row.update(in_calcite=before["calcite"], in_dolomite=before["dolomite"])
            row.update({f"out_{c}": after[c] for c in COMPONENTS})
            row.update(out_calcite=after["calcite"], out_dolomite=after["dolomite"], out_pH=after["pH"])
            rows.append(pd.DataFrame(row))
    df = pd.concat(rows, ignore_index=True)

    # sanity check: Ca, Mg and C must be conserved by the chemistry (water = 1 L per cell)
    ca = (df.in_Ca + df.in_calcite + df.in_dolomite) - (df.out_Ca + df.out_calcite + df.out_dolomite)
    mg = (df.in_Mg + df.in_dolomite) - (df.out_Mg + df.out_dolomite)
    print(f"mass balance, max abs error: Ca {ca.abs().max():.1e}  Mg {mg.abs().max():.1e} (mol/L)")
    out = HERE / "data" / "engesgaard_columns.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {out}: {len(df)} rows")


if __name__ == "__main__":
    main()
