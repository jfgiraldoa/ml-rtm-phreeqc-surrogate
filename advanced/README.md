# Advanced material (not needed for the basics)
Run the `basic/` steps 1, 2 and 4 first for the hybrid test. Outputs of these scripts are not stored in git.

- `data_generation/generate_dataset.py <config>`: random-mixing chains ("mix") and transport-like chains ("front"),
  for the Engesgaard 1D case (`config_engesgaard1d_v2.json`, 32k rows) and a 2D equilibrium case (`config_eq2d.json`).
  Front chains use upwind advection + implicit diffusion; the benchmark is Courant 1.0, diffusion number 1.339.
  Writes to `advanced/data/raw/`.
- `data_generation/one_cell_phreeqcrm.py`: a three-cell sanity check of the PhreeqcRM set-up.
- `models/train_eval.py`: decision tree, random forest and XGBoost on the change per step (6 raw inputs, no log10
  columns); split by cell and by a held-out front chain. XGBoost uses `subsample=1.0` (0.6 broke the transport loop).
- `models/hybrid_test.py`: PHREEQC only in the "active" cells and a trained model (or "no reaction") in the others.
  Most of the saving comes from not calling PHREEQC in untouched cells, which needs no model (43 % of the cells, error
  about 0). With the random forest in the inactive cells the active set can shrink to 30 % of the cells at less than 1 %
  error (with "no reaction" in the inactive cells it is 10-13 %). It measures accuracy only: PHREEQC runs in all cells
  and is used for the active ones, so the speed gain is not measured.
