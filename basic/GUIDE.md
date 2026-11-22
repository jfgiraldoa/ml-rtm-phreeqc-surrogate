# Guide to the code: what each file does and where to change things

Line numbers refer to this version of the files; use the names (functions, variables) if they drift.

## The big picture
A reactive-transport code repeats two operations at every time step:

1. **Transport** moves the dissolved totals (Ca, C, Cl, Mg) from cell to cell.
2. **Chemistry** lets PHREEQC bring every cell to equilibrium (calcite) or to its kinetic rate (dolomite).

Chemistry is called in every cell at every step, so it is the expensive part. This project asks: *can a model learn
the chemistry step and replace PHREEQC?*

```
01_generate_data.py ──► data/engesgaard_columns.csv ──► 02_train_models.py ──► results/RandomForest.joblib
   (PHREEQC does the chemistry)     (one row = one cell, one step)    (learns the chemistry)         │
                                                                                                    ▼
                                    03_plots.ipynb ◄── results/ ◄── 04_profile_with_models.py (forest inside a transport loop)
                                    (figures)                         common.py is used by 02 and 04
```

Vocabulary used below:
- **Courant number** `v·dt/dx`: how many cells the water moves per step (1.0 = exactly one cell).
- **Diffusion number** `D·dt/dx²`: how strongly the front is smoothed per step.
- **in_ / out_**: the state of a cell before the chemistry step (after transport) and after it.
- **Change over the step** (`out − in`): what PHREEQC actually computes, and what the model learns.

---

## 01_generate_data.py — make the training data with PHREEQC
**Idea:** simulate small 1D columns with a simple transport scheme and PHREEQC chemistry, and record every cell
before and after the chemistry. No machine learning here.

| Lines | What it is | Change it to... |
|---|---|---|
| 24-30 | Parameters: porosity, `DT` (533 s), `N_CELLS` (100), `N_STEPS` (40), `INLET` (injected water), `BENCHMARK`, `COLUMNS` | More or fewer steps; more flow speeds/dispersions in `COLUMNS`; a different injected water in `INLET` |
| 32-88 | `PHREEQC_INPUT` (chemistry: initial water, injected water, dolomite rate, calcite equilibrium) and `SELECTED_OUTPUT` (what we read back: calcite, dolomite, pH) | A different mineral system. Then also update `COMPONENTS` (line 25) and the features in `common.py` |
| 91-139 | `class Column`: wraps PhreeqcRM. `state()` reads the cells, `react(totals)` sets new Ca, C, Cl, Mg and runs one chemistry step | Rarely. Key point: only Ca, C, Cl, Mg are replaced; H, O and the minerals stay inside the cell |
| 142-157 | `transport()`: upwind advection, then implicit diffusion; inlet water enters at cell 0 | The transport scheme (it is deliberately simple) |
| 160-186 | `main()`: for each column and step: `before = state()` → `totals = transport(before)` → `react(totals)` → `after = state()` → save one row per cell | How rows are recorded; extra columns |

**What it writes:** `data/engesgaard_columns.csv`, 44 000 rows = 11 columns × 100 cells × 40 steps. Columns:
`column, step, cell`, then `in_Ca … in_dolomite` (what the cell receives) and `out_Ca … out_pH` (what PHREEQC returns).
It also prints a mass-balance check (Ca and Mg must be conserved; error ≈ 1e-15).

**Why the benchmark is left out (line 29-30):** the Engesgaard benchmark (Courant 1.0, diffusion 1.339) is the final
exam in step 4, so it must not be in the training data.

---

## common.py — how the model inputs are built
One function, `make_features` (lines 5-13): the 6 raw numbers of a cell (Ca, C, Cl, Mg, calcite, dolomite) plus
**log10 of the four totals**.

**Why log10?** Concentrations span 1e-10 to 1e-3, and the pH depends on log10(C). A forest cuts between training values
on the linear scale, so without the log columns it cannot tell 1e-8 from 1e-7 and the pH shows a flat plateau in the
inlet cells. Try deleting line 13's log part and re-run steps 2 and 4.

Both `02` and `04` call this function, so the model always sees inputs built the same way. **Add new inputs here**
(ratios, charge balance, ...), and nowhere else.

---

## 02_train_models.py — train the random forest
**Idea:** learn the map *(state of a cell before chemistry) → (how it changes)*.

| Lines | What it is | Change it to... |
|---|---|---|
| 29 | Read the CSV from step 1 | |
| 31-34 | Build `X` (inputs, via `make_features`) and `Y` (targets: change of Ca, C, Mg, calcite, dolomite, plus the pH after the step) | Other targets; other inputs |
| 36-37 | Split 80 % / 20 % **by cell** (`GroupShuffleSplit`): a cell and all its steps stay on the same side, otherwise the test is too easy | The test fraction |
| 38 | `StandardScaler` on the targets: they range from 1e-7 to 10, and without scaling the large ones dominate | |
| 41-42 | **The model** and its hyperparameters (`RandomForestRegressor`) | **Swap the model here**: `XGBRegressor`, an MLP, ... Keep the same `fit` / `predict` calls |
| 43-58 | Fit, predict on the test set, compute R² per target, save `metrics.csv`, `RandomForest.joblib`, `test_predictions.npz` | |

**Why the change and not the final value?** Dolomite barely changes in a step, so a model that says "nothing happens"
already gets R² ≈ 0.99 on the final value. The change is the part that is really computed.

**If you try XGBoost:** use `subsample=1.0`. With random subsampling (e.g. 0.6) it had the same test R² but failed badly
inside the transport loop (step 4).

---

## 04_profile_with_models.py — the harder test: the model inside a simulation
**Idea:** a good R² on test cells is not enough, because inside a simulation the errors of one step are the inputs of
the next one and they accumulate. We run the benchmark column twice: with PHREEQC (reference) and with the forest.

| Lines | What it is | Change it to... |
|---|---|---|
| 27-29 | Benchmark parameters (`gen.BENCHMARK`), plot units | Another column not used in training (e.g. Courant 0.6, diffusion 0.3) for a fresh test |
| 37-44 | **Reference run:** PHREEQC in every cell: `transport` → `react` | |
| 46-61 | **Model run:** at each step: `transport` → build features → `predict` → add the predicted change to the state → clip negatives to zero → next step | **Where to test other models or a hybrid** (e.g. PHREEQC in some cells, the model in the others) |
| 62-68 | Errors vs PHREEQC: max |error| divided by the range of the profile, and the relative L2 norm | Add the error per cell |
| 70-79 | Save the numbers and the figure | |

**Two errors on purpose:** the max error shows local problems (a spike at the front); the L2 norm averages them away.
Always look at both and at the figure.

**Clipping (line 57-59):** the model sometimes predicts a slightly negative concentration; we set it to zero and print
how many times it happened.

---

## 05_limits.py — where the model stops working
**Idea:** the forest learned the chemistry of one system. Here the benchmark is run again, changing one condition at a
time, with PHREEQC and with the forest. The table printed at the end is the whole point of the exercise.

| Case | What to expect |
|---|---|
| Same boundary (control) | small error (about 1-3 %) |
| Other flow speed and dispersion | still small: the model only sees the state of a cell |
| Boundary 0.5x or 3x | large error (20-77 %): new mixtures the model never saw |
| Time step 2x | dolomite fails (about 50 %): `dt` is not an input and dolomite is kinetic |

**Where to modify:** the list `CASES` (lines 32-38): change the inlet water, `dt`, Courant or diffusion number and see
what breaks. A surrogate that works in other conditions has to be trained over many of them, with the changing values
(`dt`, boundary composition, ...) added as inputs in `common.py`.

---

## 03_plots.ipynb — figures with explanations
Run it after steps 1, 2 and 4. It only reads results; it does not train anything.
Cell 3: examples of training columns. Cell 5: predicted vs PHREEQC on test cells. Cell 7: the benchmark with the forest.
It saves `results/profiles.png` and `results/parity.png`. The last cell lists experiments.

---

## Experiments to understand the code (in order)
1. Run the four steps and read the printed numbers; open the notebook.
2. In `common.py`, remove the log10 columns, re-run `02` and `04`: watch the pH plateau at x < 0.06 m.
3. In `02`, change `n_estimators` or `max_depth`: what happens to R² and to the step-4 errors?
4. In `02`, swap the forest for XGBoost with `subsample=0.6`, then `1.0`: same R², very different step-4 result.
5. In `01`, add columns to `COLUMNS` and re-run everything: do the errors fall?
6. In `04`, change the benchmark to a column that was not in training: is the error the same?
7. Run `05_limits.py`, then change `CASES` (e.g. a boundary 0.8x, or a time step 1.1x): where does it start to fail?
8. Put the forest only in the cells where `Mg < 1e-6` and PHREEQC elsewhere (a hybrid): how much accuracy do you keep,
   and in what fraction of the cells do you still call PHREEQC?
