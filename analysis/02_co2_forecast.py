"""Analysis 2: forecasting CO2 from January 1998 to December 2001 (48 months), the design of version 1.

Why: version 1 forecast the 48 test months with ARIMA(1,1,1) and with ARIMA(2,1,3), the order chosen by its AIC
search. Neither model has a seasonal part, and statsmodels' ARIMA has no constant when d = 1, so the forecasts
of both models level off: they can follow neither the cycle nor the trend.

Design: every model is estimated once on the training sample and forecast 1 to 48 months ahead from December 1997.
* Version 1's two models, on version 1's data (from 1958, missing months filled with the next month).
* On the data of version 2 (from 1965, no filled values): seasonal naive with drift, additive Holt-Winters (version 1
  imported it but did not use it), and the seasonal ARIMA chosen in analysis 1.
* The same seasonal ARIMA on version 1's data, to show how much the choice of sample matters.
* Table 8: MAE, RMSE, mean error, and the share of months inside the 95% interval where the model gives one.
* Figure 3: the forecasts against the data, one panel per model.
"""

import numpy as np
import pandas as pd

from common import save_table
from armaco2.data import TEST_START, TRAIN_END, co2_monthly, co2_v1, co2_v2, split
from armaco2.models import accuracy, fit_arima_v1, fit_sarima, forecast_frame, holt_winters, seasonal_naive_drift
from armaco2.style import BAND, BLUE, INK, decimal_axis, event_line, panel_name, save_figure, source_note, \
    use_style

y1, y2 = co2_v1(), co2_v2()
tr1, te1 = split(y1)
tr2, te2 = split(y2)
assert te1.equals(te2)                      # the test months have no missing samples: same data in both versions
test, H, idx = te2, len(te2), te2.index

chosen = pd.read_csv("../figures/table4_chosen_model.csv").set_index("term").loc["order", "estimate"]
order, seas = (0, 1, 1), (0, 1, 1, 12)
assert chosen == f"{order}{seas}", chosen   # analysis 1 chose the airline model

fc = {}
CONVERGED = {}
for name, o in (("ARIMA(1,1,1), version 1", (1, 1, 1)), ("ARIMA(2,1,3), version 1", (2, 1, 3))):
    res_v1 = fit_arima_v1(tr1, o)
    CONVERGED[name] = bool(res_v1.mle_retvals.get("converged", True))
    fc[name] = forecast_frame(res_v1, H, idx)
fc["Seasonal naive with drift"] = seasonal_naive_drift(tr2, H, idx)
fc["Holt-Winters, additive"] = holt_winters(tr2, H, idx)
fc["SARIMA(0,1,1)(0,1,1)$_{12}$"] = forecast_frame(fit_sarima(tr2, order, seas), H, idx)
fc["SARIMA, version 1's data"] = forecast_frame(fit_sarima(tr1, order, seas), H, idx)

rows, out = [], pd.DataFrame({"actual": test})
for name, f in fc.items():
    rows.append({"model": name.replace("$_{12}$", "_12"), **accuracy(test, f), "converged": CONVERGED.get(name, "")})
    for c in f:
        out[f"{name.replace('$_{12}$', '_12')}|{c}"] = f[c]
save_table(rows, "table8_forecast_accuracy")
out.to_csv("../figures/co2_forecasts.csv", float_format="%.6f")
for r in rows:
    print(f"{r['model']:32s} MAE {r['mae']:.3f} RMSE {r['rmse']:.3f} bias {r['bias']:+.3f} cover {r.get('coverage95', np.nan):.3f}")

# the airline model on three samples: version 2 (from 1965), version 1 (from 1958, filled months), and from 1958
# without filling (the Kalman filter skips the months without a sample)
samples = {"1965-1997, version 2": tr2, "1958-1997, filled (version 1)": tr1,
           "1958-1997, missing months skipped": co2_monthly()[:TRAIN_END]}
srows = []
for label, s in samples.items():
    res = fit_sarima(s, order, seas)
    srows.append({"sample": label, "theta": float(res.params["ma.L1"]), "Theta": float(res.params["ma.S.L12"]),
                  "sigma2": float(res.params["sigma2"]),
                  "mae": accuracy(test, forecast_frame(res, H, idx))["mae"]})
save_table(srows, "table8b_sarima_samples")

# errors of the seasonal ARIMA by year
e = fc["SARIMA(0,1,1)(0,1,1)$_{12}$"]["mean"] - test
save_table([{"year": int(yr), "mean_error": float(v.mean()), "mae": float(v.abs().mean())}
            for yr, v in e.groupby(e.index.year)], "table9_sarima_errors_by_year")

use_style()
SHOW_FROM = "1995-01-01"
NAMES = {"ARIMA(1,1,1), version 1": "ARIMA(1,1,1)\nversion 1", "ARIMA(2,1,3), version 1": "ARIMA(2,1,3)\nversion 1",
         "Seasonal naive with drift": "Seasonal naive\nwith drift", "Holt-Winters, additive": "Holt-Winters\nadditive",
         "SARIMA(0,1,1)(0,1,1)$_{12}$": "SARIMA\n(0,1,1)(0,1,1)$_{12}$",
         "SARIMA, version 1's data": "SARIMA, same\nmodel, v1 data"}


def draw3():
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(4.75, 3.35), constrained_layout=True, sharex=True, sharey=True)
    hist = y2[SHOW_FROM:]
    for ax, (name, f) in zip(axes.flat, fc.items()):
        if "lo" in f:
            ax.fill_between(f.index, f["lo"], f["hi"], color=BAND, lw=0, label="95% interval")
        ax.plot(hist.index, hist.values, color=INK, lw=1.0, label="Data")
        ax.plot(f.index, f["mean"], color=BLUE, lw=1.4, label="Forecast")
        event_line(ax, pd.Timestamp(TEST_START), "Dec\n1997", ha="right")
        a = accuracy(test, f)
        ax.text(0.98, 0.99, f"MAE {a['mae']:.2f}", transform=ax.transAxes, va="top", ha="right", fontsize=7.5,
                fontweight="bold", color=BLUE)
        panel_name(ax, NAMES[name], x=0.97, y=0.04)
        ax.set_ylim(352, 380)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        decimal_axis(ax)
    axes[0, 0].set_ylabel("ppm")
    axes[1, 0].set_ylabel("ppm")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7.5)
    return fig


save_figure("fig3_co2_forecasts", draw3,
            lambda: "Figure 3. Forecasts of CO$_2$ for 1998-2001 made in December 1997, by model")
