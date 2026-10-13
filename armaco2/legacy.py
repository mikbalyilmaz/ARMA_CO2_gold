"""Runs the code of version 1 (legacy/ARMA_CO2_forecast_v1.ipynb) unchanged, for comparison.

The code cells are executed in order in one namespace, with figures drawn to a non-interactive backend and printed
output captured. The notebook's first ARIMA(1,1,1) forecast is kept as `y_pred_111` before later cells overwrite
`y_pred`.

Author: Muhammed Ikbal Yilmaz
"""

import contextlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parents[1] / "legacy" / "ARMA_CO2_forecast_v1.ipynb"


def code_cells(path=NOTEBOOK):
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def run_v1():
    """Executes all code cells of version 1; returns (namespace, printed output)."""
    ns, out = {}, io.StringIO()
    with contextlib.redirect_stdout(out):
        for i, src in enumerate(code_cells()):
            exec(compile(src, f"<v1 cell {i}>", "exec"), ns)      # noqa: S102  (trusted file of this repository)
            if i == 2:                                             # the ARIMA(1,1,1) cell
                ns["y_pred_111"] = ns["y_pred"].copy()
            plt.close("all")
    return ns, out.getvalue()
