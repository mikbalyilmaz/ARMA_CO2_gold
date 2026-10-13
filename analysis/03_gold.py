"""Analysis 3: the monthly gold price, the second series of version 1.

Why: version 1 plotted a monthly gold price with "7-day" and "30-day" moving averages and an exponential smoothing,
and a histogram of the price with a fitted normal density. The data are monthly, so the windows are 7 and 30 months;
the price is in US dollars per troy ounce, not pounds; and a histogram of a trending price level does not describe
a distribution. The time-series question behind it is the one of the course: is there anything in the past of the
gold price that an ARMA model can use?

Data: gold closing price on Borsa Istanbul, US dollars per troy ounce, last close of each month, July 2018 to
September 2026 (CBRT EVDS); r_t = 100 (ln P_t - ln P_{t-1}). Version 1's own file (from March 2012) is not in its
repository, and EVDS has this series from July 2018.
* Table 10: ADF and KPSS on ln P_t and on r_t.
* Table 11: ARMA(p,q) with a constant for r_t, p, q in 0..2, ranked by AIC; Ljung-Box on r_t and on r_t^2.
* Table 12: one-month-ahead forecasts over the last 36 months, re-estimated every month on all earlier months:
  historical mean, zero (random walk in the log price), and the ARMA order chosen by AIC on the first estimation
  window; out-of-sample R^2 against the historical mean.
* Figure 4: (a) the price with version 1's smoothers on a log scale; (b) autocorrelations of r_t;
  (c) the distribution of r_t against the normal density; (d) cumulative squared-error difference of the forecasts.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf

from common import save_table
from armaco2.data import load_gold
from armaco2.models import ljung_box, unit_root
from armaco2.style import BAND, BLUE, INK, RED, decimal_axis, end_label, save_figure, source_note, \
    use_style

OOS = 36

gold = load_gold()
p = gold
lp = np.log(p)
r = (100 * lp.diff()).dropna()

# Table 10
save_table([{"series": "log price", "deterministic": "ct", **unit_root(lp, "ct")},
            {"series": "monthly log return", "deterministic": "c", **unit_root(r, "c")}], "table10_gold_unit_roots")


def fit_arma(x, p_, q_):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(x.values, order=(p_, 0, q_), trend="c").fit()


# Table 11
rows = []
for p_, q_ in itertools.product(range(3), range(3)):
    res = fit_arma(r, p_, q_)
    rows.append({"p": p_, "q": q_, "aic": res.aic, "bic": res.bic})
sel = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
sel.to_csv("../figures/table11_gold_arma_selection.csv", index=False, float_format="%.4f")
lb = ljung_box(r, [6, 12, 24])
lb2 = ljung_box(r ** 2, [6, 12, 24])
jb = stats.jarque_bera(r)
facts = {"first": f"{p.index[0]:%Y-%m}", "last": f"{p.index[-1]:%Y-%m}", "months": len(p), "returns": len(r),
         "price_first": float(p.iloc[0]), "price_last": float(p.iloc[-1]), "price_min": float(p.min()),
         "price_min_month": f"{p.idxmin():%Y-%m}", "mean_r": float(r.mean()), "sd_r": float(r.std()),
         "skew_r": float(stats.skew(r)), "exkurt_r": float(stats.kurtosis(r)), "jb_stat": float(jb.statistic),
         "jb_p": float(jb.pvalue), "aic_best_p": int(sel.loc[0, "p"]), "aic_best_q": int(sel.loc[0, "q"]),
         "aic_best": float(sel.loc[0, "aic"]),
         "aic_00": float(sel.loc[(sel.p == 0) & (sel.q == 0), "aic"].iloc[0]),
         "bic_best_p": int(sel.sort_values("bic").iloc[0]["p"]), "bic_best_q": int(sel.sort_values("bic").iloc[0]["q"])}
for row in lb.itertuples():
    facts[f"lb_r_{int(row.lag)}"], facts[f"lb_r_p_{int(row.lag)}"] = float(row.lb_stat), float(row.lb_pvalue)
for row in lb2.itertuples():
    facts[f"lb_r2_{int(row.lag)}"], facts[f"lb_r2_p_{int(row.lag)}"] = float(row.lb_stat), float(row.lb_pvalue)

# Table 12: one-month-ahead forecasts, expanding window; every month the ARMA order is chosen again by AIC
# (p, q in 0..2) on the months before, and the chosen model is re-estimated
fc_rows = []
for t in range(len(r) - OOS, len(r)):
    hist = r.iloc[:t]
    fits = {(p_, q_): fit_arma(hist, p_, q_) for p_, q_ in itertools.product(range(3), range(3))}
    (po_t, qo_t), res = min(fits.items(), key=lambda kv: kv[1].aic)
    fc_rows.append({"month": f"{r.index[t]:%Y-%m}", "actual": float(r.iloc[t]), "mean": float(hist.mean()),
                    "zero": 0.0, "arma": float(res.forecast(1)[0]), "p": po_t, "q": qo_t})
fcdf = pd.DataFrame(fc_rows)
fcdf.to_csv("../figures/gold_forecasts.csv", index=False, float_format="%.6f")
sse = {k: float(((fcdf["actual"] - fcdf[k]) ** 2).sum()) for k in ("mean", "zero", "arma")}
facts.update({"oos_first": fcdf["month"].iloc[0], "oos_last": fcdf["month"].iloc[-1], "oos_months": OOS,
              "oos_orders": " ".join(f"({a},{b}):{n}" for (a, b), n in
                                     fcdf.groupby(["p", "q"]).size().items()),
              "oos_months_not_mean": int(((fcdf["p"] + fcdf["q"]) > 0).sum()),
              "rmse_mean": np.sqrt(sse["mean"] / OOS), "rmse_zero": np.sqrt(sse["zero"] / OOS),
              "rmse_arma": np.sqrt(sse["arma"] / OOS), "r2os_arma": 1 - sse["arma"] / sse["mean"],
              "r2os_zero": 1 - sse["zero"] / sse["mean"]})
save_table([facts], "table12_gold_facts")
print({k: facts[k] for k in ("first", "last", "aic_best_p", "aic_best_q", "lb_r_p_12", "lb_r2_p_12", "r2os_arma",
                             "r2os_zero", "jb_p")})

ma7, ma30 = p.rolling(7).mean(), p.rolling(30).mean()
ewm = p.ewm(span=30, adjust=False).mean()
LAGS = 24
ac = acf(r, nlags=LAGS, fft=True)
save_table([{"lag": k, "acf": float(ac[k])} for k in range(1, LAGS + 1)], "table13_gold_acf")

use_style()


def draw4():
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    fig, axes = plt.subplots(2, 2, figsize=(4.75, 3.45), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(p.index, p.values, color=INK, lw=1.4)
    ax.plot(ma7.index, ma7.values, color=BLUE, lw=1.1, ls="--")
    ax.plot(ma30.index, ma30.values, color=BLUE, lw=2.0)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.FixedLocator([1000, 1500, 2000, 3000, 5000, 8000]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylim(900, 9000)
    end_label(ax, p.index[-1], p.iloc[-1], f"{p.iloc[-1]:,.0f}", INK, dx=3)
    ax.text(0.03, 0.97, "US\\$ per ounce, log scale\nLast close of the month,\nBorsa Istanbul (CBRT EVDS)\n"
            "Dashed: 7-month mean\nThick blue: 30-month mean", transform=ax.transAxes, va="top", fontsize=6.4)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("(a) Gold price")
    ax = axes[0, 1]
    band = 1.96 / np.sqrt(len(r))
    ax.axhspan(-band, band, color="#e6e6e6", lw=0, zorder=0)
    ax.vlines(range(1, LAGS + 1), 0, ac[1:], color=BLUE, lw=1.8)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(-0.45, 0.45)
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_xlabel("Lag (months)")
    ax.set_title("(b) ACF of monthly returns")
    lbp = facts["lb_r_p_12"]
    lbtxt = "p < 0.01" if lbp < 0.01 else f"p = {lbp:.2f}"
    source_note(ax, f"Ljung-Box Q(12): {lbtxt}\nGrey band: $\\pm 1.96/\\sqrt{{n}}$", y=0.03)
    decimal_axis(ax)
    ax = axes[1, 0]
    ax.hist(r.values, bins=20, density=True, color=BAND, edgecolor="white", lw=0.4)
    xs = np.linspace(r.min() - 2, r.max() + 2, 300)
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color=INK, lw=1.6)
    ax.axvline(r.mean(), color=RED, ls=(0, (1, 1.4)), lw=1.6)
    ax.set_xlabel("Monthly log return, %")
    ax.set_title("(c) Distribution of returns")
    source_note(ax, f"Mean {r.mean():.2f}%\ns.d. {r.std():.2f}%\nBlack: normal\ndensity", x=0.98, y=0.97,
                ha="right", va="top")
    decimal_axis(ax)
    decimal_axis(ax, "x")
    ax = axes[1, 1]
    idx = pd.to_datetime(fcdf["month"] + "-01")
    base = (fcdf["actual"] - fcdf["mean"]) ** 2
    for key, col, lab in (("zero", INK, "Zero"), ("arma", BLUE, "ARMA")):
        cum = (base - (fcdf["actual"] - fcdf[key]) ** 2).cumsum()
        ax.plot(idx, cum.values, color=col, lw=1.8)
        end_label(ax, idx.iloc[-1], cum.iloc[-1], lab, col, dx=3)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title("(d) Forecasts against the mean")
    ax.set_xticks([pd.Timestamp(f"{yr}-01-01") for yr in (2024, 2025, 2026)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(idx.iloc[0] - pd.Timedelta(days=20), idx.iloc[-1] + pd.Timedelta(days=330))
    ax.spines["bottom"].set_bounds(mdates.date2num(idx.iloc[0]), mdates.date2num(idx.iloc[-1]))
    decimal_axis(ax)
    return fig


save_figure("fig4_gold", draw4, lambda: f"Figure 4. Monthly gold price, {p.index[0]:%B %Y} to {p.index[-1]:%B %Y}")
