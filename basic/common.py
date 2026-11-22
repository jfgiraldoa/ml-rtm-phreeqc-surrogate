"""Shared by steps 2 and 4: how the model inputs are built from the state of a cell."""
import numpy as np


def make_features(ca, c, cl, mg, calcite, dolomite):
    """6 raw inputs + log10 of the 4 dissolved totals.

    The log10 columns matter: concentrations span many decades (1e-10 ... 1e-3) and the pH depends on log10(C), but a
    forest places its splits between training values on the linear scale, so without them it cannot tell 1e-8 from
    1e-7 and returns a flat pH plateau in the inlet cells.
    """
    raw = np.column_stack([ca, c, cl, mg, calcite, dolomite])
    return np.column_stack([raw, np.log10(np.clip(raw[:, :4], 1e-12, None))])
