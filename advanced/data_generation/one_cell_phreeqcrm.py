"""Minimal check: equilibrate a few cells with PhreeqcRM, same setup as a PhreeqcRM 2D coupling.

Each cell i gets its own EQUILIBRIUM_PHASES i (calcite, dolomite amounts in mol/L water), then the
aqueous totals (mol/L water) are set and one equilibrium step is run. No transport.
"""
from pathlib import Path

import numpy as np
from phreeqcrm import PhreeqcRM

DB = Path(__file__).parent / "database" / "phreeqc.dat"
COMPONENTS = ("Ca", "C", "Cl", "Mg")
POROSITY = 0.3

# (Ca, C, Cl, Mg) in mol/L water, calcite, dolomite in mol/L water
CASES = [
    (1.22e-4, 1.22e-4, 1e-12, 1e-12, 2.14e-5, 0.0),   # initial pore water of the 2D case
    (1e-10, 1e-10, 2e-3, 1e-3, 2.14e-5, 0.0),         # inlet solution on that mineral
    (6e-4, 5e-4, 1e-3, 5e-4, 2.14e-5, 1e-5),          # an intermediate (mixed) state
]


def equilibrate(cases):
    n = len(cases)
    rm = PhreeqcRM(n, 1)
    rm.SetErrorHandlerMode(0)
    rm.SetScreenOn(False)
    assert rm.LoadDatabase(str(DB)) == 0
    rm.SetUnitsSolution(2)        # mol/L
    rm.SetUnitsPPassemblage(1)    # input mol/L water
    rm.SetPorosity(np.full(n, POROSITY))
    rm.SetSaturationUser(np.ones(n))
    rm.SetRepresentativeVolume(np.full(n, 1.0 / POROSITY))
    rm.UseSolutionDensityVolume(False)

    phases = "\n".join(
        f"EQUILIBRIUM_PHASES {i + 1}\n    Calcite 0 {c[4]:.8e}\n    Dolomite 0 {c[5]:.8e}"
        for i, c in enumerate(cases)
    )
    rm.RunString(True, True, True, f"""
SOLUTION 10 pore water
    temp 25
    pH 7 charge
    pe 4
    units mol/kgw
    Ca 1.22e-4
    C 1.22e-4
    Cl 1e-12
    Mg 1e-12
{phases}
END
""")
    rm.RunString(True, False, False, """
SELECTED_OUTPUT 1
    -reset false
    -high_precision true
USER_PUNCH 1
    -headings Calcite Dolomite pH SI_Calcite SI_Dolomite
    10 PUNCH EQUI("Calcite"), EQUI("Dolomite"), -LA("H+"), SI("Calcite"), SI("Dolomite")
END
""")
    rm.SetSelectedOutputOn(True)
    assert rm.FindComponents() > 0
    comps = [str(c) for c in rm.GetComponents()]
    ic = np.full(7 * n, -1, dtype=np.int32)
    ic[0:n] = 10
    ic[n:2 * n] = np.arange(1, n + 1)
    assert rm.InitialPhreeqc2Module(ic) == 0
    rm.SetTimeStep(0.0); rm.SetTime(0.0)
    rm.RunCells()

    conc = np.asarray(rm.GetConcentrations()).reshape(len(comps), n).copy()
    for name in COMPONENTS:
        conc[comps.index(name), :] = [c[COMPONENTS.index(name)] for c in cases]
    rm.SetConcentrations(conc.reshape(-1))
    rm.SetTimeStep(1.0); rm.SetTime(1.0)
    rm.RunCells()

    out = np.asarray(rm.GetConcentrations()).reshape(len(comps), n)
    rm.SetCurrentSelectedOutputUserNumber(1)
    heads = [str(h) for h in rm.GetSelectedOutputHeadings()]
    sel = np.asarray(rm.GetSelectedOutput()).reshape(len(heads), n)
    return comps, out, dict(zip(heads, sel))


if __name__ == "__main__":
    comps, out, sel = equilibrate(CASES)
    for i, c in enumerate(CASES):
        print(f"cell {i}: in  Ca,C,Cl,Mg={c[:4]} calcite={c[4]:.3e} dolomite={c[5]:.3e}")
        print("        out", {k: f"{out[comps.index(k), i]:.4e}" for k in COMPONENTS},
              {k: f"{v[i]:.4e}" for k, v in sel.items()})
