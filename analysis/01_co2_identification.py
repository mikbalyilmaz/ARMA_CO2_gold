"""Analysis 1: identifying a model for monthly CO2 at Mauna Loa (the Box-Jenkins steps).

Why: version 1 fitted non-seasonal ARIMA models to a series whose most visible feature is a twelve-month cycle, and
chose the order by comparing AIC across differencing orders d = 0..3, which compares likelihoods of different data.

Design: training sample of version 2, January 1965 to December 1997 (396 months, no month without a sample).
* Table 1: ADF and KPSS tests on the level, the first difference, the seasonal difference and both.
* Table 2: version 1's search re-run: the smallest AIC for each d, to show why AIC cannot choose d.
* Table 3: seasonal ARIMA(p,1,q)(P,1,Q)_12, p, q in 0..2, P, Q in 0..1, ranked by AIC (all on the same data).
* Table 4: the chosen model: estimates, standard errors, Ljung-Box tests of its residuals.
* Figure 1: the series, its twelve-month change, and its seasonal cycle.
* Figure 2: autocorrelations and partial autocorrelations of the four transformations.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

from common import save_table
from armaco2.data import TEST_START, co2_monthly, co2_v1, co2_v2, split
from armaco2.models import arima_grid_v1, fit_sarima, ljung_box, select_sarima, transforms, unit_root
from armaco2.style import BLUE, INK, MUTED, decimal_axis, end_label, event_line, panel_name, save_figure, \
    source_note, use_style

y_all = co2_monthly()
y = co2_v2()
train, test = split(y)

# Table 1
rows = []
for name, (x, trend) in transforms(train).items():
    r = unit_root(x, trend)
    rows.append({"series": name, "deterministic": trend, **r})
save_table(rows, "table1_unit_roots")

# Table 2: version 1's grid on version 1's data, best AIC for each d
v1_train, _ = split(co2_v1())
grid = arima_grid_v1(v1_train)
grid.to_csv("../figures/table2_v1_grid_all.csv", index=False, float_format="%.4f")
best = grid.dropna().loc[grid.dropna().groupby("d")["aic"].idxmin()]
save_table([{"d": int(r.d), "best_p": int(r.p), "best_q": int(r.q), "aic": float(r.aic),
             "observations_after_differencing": len(v1_train) - int(r.d)} for r in best.itertuples()],
           "table2_v1_aic_by_d")

# Table 3
sel = select_sarima(train)
sel.to_csv("../figures/table3_sarima_selection.csv", index=False, float_format="%.4f")
top = sel.iloc[0]
order, seas = (int(top.p), 1, int(top.q)), (int(top.P), 1, int(top.Q), 12)
print("chosen", order, seas, "AIC", round(top.aic, 3))

# Table 4
res = fit_sarima(train, order, seas)
k = int(top.p + top.q + top.P + top.Q)
lb = ljung_box(res.resid[13:], [12, 24, 36], fitted_params=k)    # the first 13 residuals are start-up values
rows = [{"term": n, "estimate": float(res.params[n]), "se": float(res.bse[n])} for n in res.params.index]
rows += [{"term": f"LjungBox_Q{int(r.lag)}", "estimate": float(r.lb_stat), "se": float(r.lb_pvalue)}
         for r in lb.itertuples()]
rows += [{"term": "AIC", "estimate": float(res.aic)}, {"term": "n", "estimate": float(res.nobs)},
         {"term": "order", "estimate": f"{order}{seas}"}]
save_table(rows, "table4_chosen_model")

# Figure inputs
dec = seasonal_decompose(train, model="additive", period=12)        # centred 2x12 moving average for the trend
cycle = dec.seasonal.iloc[:12].groupby(dec.seasonal.index[:12].month).first().sort_index()
growth = y_all.diff(12)
save_table([{"month": int(m), "seasonal_effect_ppm": float(v)} for m, v in cycle.items()]
           + [{"month": "amplitude", "seasonal_effect_ppm": float(cycle.max() - cycle.min())}], "table5_seasonal_cycle")
save_table([{"first": str(y_all.index[0].date()), "last": str(y_all.index[-1].date()), "months": len(y_all),
             "missing_months": int(y_all.isna().sum()),
             "missing_list": " ".join(d.strftime("%Y-%m") for d in y_all[y_all.isna()].index),
             "first_value": float(y_all.dropna().iloc[0]), "last_value": float(y_all.iloc[-1]),
             "growth_1965_1974": float(growth["1965":"1974"].mean()), "growth_1988_1997": float(growth["1988":"1997"].mean()),
             "growth_1998": float(growth["1998-12"].iloc[0]), "train_months": len(train), "test_months": len(test)}],
           "table6_co2_facts")

use_style()


def draw1():
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4.75, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.05, 1])
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax = axes[0]
    ax.plot(y_all.index, y_all.values, color=INK, lw=1.1)
    event_line(ax, pd.Timestamp(TEST_START), "Jan\n1998", y=0.30, ha="right")
    end_label(ax, y_all.index[-1], y_all.iloc[-1], f"{y_all.iloc[-1]:.1f}", INK, dx=3)
    ax.set_title("(a) CO$_2$ at Mauna Loa, ppm")
    source_note(ax, "Monthly mean of weekly flask samples\n(Source: Keeling and Whorf, SIO/CDIAC)",
                x=0.02, y=0.97, va="top")
    ax = axes[1]
    ax.plot(growth.index, growth.values, color=BLUE, lw=1.0)
    ax.axhline(0, color=MUTED, lw=0.8)
    event_line(ax, pd.Timestamp(TEST_START), "Jan\n1998", ha="right")
    means = []
    for a, b in (("1965", "1974"), ("1988", "1997")):
        m = growth[a:b].mean()
        means.append(f"{a}-{b[2:]}: {m:.2f}")
        ax.plot([pd.Timestamp(f"{a}-01-01"), pd.Timestamp(f"{b}-12-01")], [m, m], color=INK, lw=2.4)
    ax.text(0.02, 0.98, "Black bars: average change\n" + "; ".join(means), transform=ax.transAxes, va="top",
            fontsize=6.6, fontweight="bold")
    ax.set_ylim(-0.6, 5.6)
    ax.set_title("(b) 12-month change, ppm")
    ax = axes[2]
    months = np.arange(1, 13)
    ax.bar(months, cycle.values, color=BLUE, width=0.7)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(months, list("JFMAMJJASOND"))
    ax.set_title("(c) Seasonal cycle, ppm")
    source_note(ax, "Deviation from a\ncentred 12-month\nmean, 1965-97", x=0.02, y=0.03)
    for a in axes[:2]:
        a.xaxis.set_major_locator(mdates.YearLocator(10))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a in axes:
        decimal_axis(a)
    return fig


save_figure("fig1_co2", draw1, lambda: "Figure 1. Monthly CO$_2$ at Mauna Loa: trend, growth and seasonal cycle")

LAGS = 36
panels = [(name, x.dropna()) for name, (x, _) in transforms(train).items()]
LABEL = {"level": "$y_t$", "diff": r"$\Delta y_t$", "seasonal diff": r"$\Delta_{12} y_t$",
         "diff and seasonal diff": r"$\Delta\Delta_{12} y_t$"}
acf_rows = []
for name, x in panels:
    a, p = acf(x, nlags=LAGS, fft=True), pacf(x, nlags=LAGS, method="ywmle")
    for lag in range(1, LAGS + 1):
        acf_rows.append({"series": name, "lag": lag, "acf": float(a[lag]), "pacf": float(p[lag]), "n": len(x)})
save_table(acf_rows, "table7_acf")


def draw2():
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(4.75, 2.65), constrained_layout=True, sharex=True, sharey=True)
    for j, (name, x) in enumerate(panels):
        band = 1.96 / np.sqrt(len(x))
        rows = [r for r in acf_rows if r["series"] == name]
        lags = [r["lag"] for r in rows]
        for i, key in enumerate(("acf", "pacf")):
            ax = axes[i, j]
            vals = [r[key] for r in rows]
            ax.vlines(lags, 0, vals, color=[BLUE if L % 12 else INK for L in lags], lw=1.3)
            ax.axhspan(-band, band, color="#e6e6e6", zorder=0, lw=0)
            ax.axhline(0, color=INK, lw=0.8)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xticks([1, 12, 24, 36])
            decimal_axis(ax)
        axes[0, j].set_title(LABEL[name], loc="center", fontsize=9)
    axes[0, 0].set_ylabel("Autocorrelation")
    axes[1, 0].set_ylabel("Partial autocorrelation")
    fig.supxlabel("Lag (months). Black bars: lags 12, 24, 36; grey band: $\\pm 1.96/\\sqrt{n}$", fontsize=7)
    return fig


save_figure("fig2_acf", draw2, lambda: "Figure 2. Autocorrelations of CO$_2$ and its differences, 1965-1997")
