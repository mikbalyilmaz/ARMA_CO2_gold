"""Unit tests. Run from the repository root:  python tests/run_tests.py

Each result is compared with an independent computation: averages written out by hand, simulated series with a
known answer, formulas written out, and the printed output of version 1.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from armaco2.data import co2_monthly, co2_v1, co2_v2, load_co2_weekly, load_gold, split  # noqa: E402
from armaco2.models import (accuracy, fit_sarima, ljung_box, seasonal_naive_drift, select_sarima,  # noqa: E402
                            unit_root)

warnings.simplefilter("ignore")
failed = 0


def check(name, ok):
    global failed
    print(f"{'ok' if ok else 'FAIL':<5}{name}")
    if not ok:
        failed += 1


# 1. Monthly CO2: mean of the weekly samples of each month; months without a sample are missing
w = load_co2_weekly()
m = co2_monthly(w)
ok = True
for month in pd.to_datetime(["1958-03-01", "1975-07-01", "1997-12-01", "2001-12-01"]):
    vals = w[(w.index.year == month.year) & (w.index.month == month.month)].dropna()
    ok &= abs(vals.mean() - m[month]) < 1e-12
ok &= [d.strftime("%Y-%m") for d in m[m.isna()].index] == ["1958-06", "1958-10", "1964-02", "1964-03", "1964-04"]
v1 = co2_v1(w)
ok &= v1["1964-02-01"] == m["1964-05-01"] and v1.notna().all()          # version 1 fills with the next month
tr, te = split(co2_v2(w))
ok &= len(tr) == 396 and len(te) == 48 and tr.index[0] == pd.Timestamp("1965-01-01") and tr.notna().all()
check("monthly means equal direct averages; 5 missing months; v1 back-fill; v2 sample 396 + 48 months", ok)

# 2. Seasonal naive with drift, written out
x = pd.Series(np.arange(36.0) + np.tile(np.arange(12.0), 3))
f = seasonal_naive_drift(x, 14, range(14))["mean"].values
g = np.mean(x.values[12:] - x.values[:-12])                               # = 12 here
ok = np.allclose(f[:12], x.values[24:36] + g) and np.allclose(f[12:14], x.values[24:26] + 2 * g) and g == 12
check("seasonal naive with drift: last year's month plus the average twelve-month change per year ahead", ok)

# 3. Accuracy measures by hand
a = np.array([1.0, 2.0, 3.0, 4.0])
fc = pd.DataFrame({"mean": [1.5, 1.0, 3.0, 6.0], "lo": [0, 0, 3.5, 0], "hi": [2, 3, 4, 7]})
r = accuracy(a, fc)
ok = abs(r["mae"] - 0.875) < 1e-12 and abs(r["rmse"] - np.sqrt((0.25 + 1 + 0 + 4) / 4)) < 1e-12
ok &= abs(r["bias"] - 0.375) < 1e-12 and r["coverage95"] == 0.75
check("MAE, RMSE, mean error and interval coverage equal direct computation", ok)

# 4. Ljung-Box statistic written out, with degrees of freedom reduced by the fitted parameters
rng = np.random.default_rng(1)
e = rng.standard_normal(300)
n = len(e)
ec = e - e.mean()
rho = np.array([np.sum(ec[k:] * ec[:-k]) / np.sum(ec ** 2) for k in range(1, 13)])
q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, 13)))
lb = ljung_box(e, [12], fitted_params=2)
from scipy import stats  # noqa: E402

ok = abs(lb["lb_stat"].iloc[0] - q) < 1e-9 and abs(lb["lb_pvalue"].iloc[0] - stats.chi2.sf(q, 10)) < 1e-12
check("Ljung-Box Q(12) and its p-value with 12 - 2 degrees of freedom equal the formula", ok)

# 5. Unit-root tests: a random walk is not rejected, white noise is; KPSS the other way round
rw = np.cumsum(rng.standard_normal(400))
wn = rng.standard_normal(400)
u_rw, u_wn = unit_root(rw, "c"), unit_root(wn, "c")
check("ADF does not reject a random walk and rejects white noise; KPSS the reverse",
      u_rw["adf_p"] > 0.05 and u_wn["adf_p"] < 0.01 and u_rw["kpss_p"] <= 0.05 and u_wn["kpss_p"] >= 0.05)

# 6. Known answer: the airline model is recovered from a simulated airline series, and AIC ranks it first
theta, Theta, n = -0.4, -0.6, 1200
eps = rng.standard_normal(n + 13)
wser = eps[13:] + theta * eps[12:-1] + Theta * eps[1:-12] + theta * Theta * eps[:-13]   # (1+tB)(1+TB^12) e
z = np.zeros(n + 13)
for t in range(13, n + 13):                                    # invert (1 - B)(1 - B^12)
    z[t] = z[t - 1] + z[t - 12] - z[t - 13] + wser[t - 13]
ys = pd.Series(z[13:], index=pd.date_range("1900-01-01", periods=n, freq="MS"))
res = fit_sarima(ys, (0, 1, 1), (0, 1, 1, 12))
sel = select_sarima(ys, p=range(2), q=range(2), P=range(2), Q=range(2))
ok = abs(res.params["ma.L1"] - theta) < 0.08 and abs(res.params["ma.S.L12"] - Theta) < 0.08
ok &= tuple(sel.loc[0, ["p", "q", "P", "Q"]]) == (0, 1, 0, 1) and abs(sel.loc[0, "aic"] - res.aic) < 1e-6
check("simulated airline series: theta and Theta recovered within 0.08; AIC selects (0,1,1)(0,1,1)_12", ok)

# 7. Version 1 reproduced: its AIC search and the MAE of its ARIMA(1,1,1) forecast
from armaco2.legacy import run_v1  # noqa: E402

ns, printed = run_v1()
v1_printed = {(0, 0, 0): 3954.01, (0, 0, 1): 3312.73, (0, 0, 2): 2771.31, (0, 0, 3): 2328.74, (0, 1, 0): 1523.91,
              (0, 1, 1): 1254.79, (0, 1, 2): 1137.97, (0, 1, 3): 1094.51, (2, 1, 1): 979.80, (2, 1, 2): 979.50}
ok = all(f"Best ARIMA{k} AIC={v:.2f}" in printed for k, v in v1_printed.items())
ok &= ns["best_params_aic"] == (2, 1, 3)
mae111 = float(np.mean(np.abs(ns["y_pred_111"].values - ns["test"].values)))
ok &= round(mae111, 1) == 2.3                                     # the title of version 1's figure: MAE=2.3
check("version 1's own code: its printed AIC path, the chosen ARIMA(2,1,3) and MAE 2.3 of ARIMA(1,1,1)", ok)

# 8. Gold: the stored series equals the EVDS export, month by month, without its incomplete last month
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from prepare_gold import read_evds  # noqa: E402

gold = load_gold()
raw = read_evds()
ok = gold.index.freqstr == "MS" and gold.notna().all() and (gold > 0).all()
ok &= gold.index[0] == pd.Timestamp("2018-07-01") and gold.index[-1] == pd.Timestamp("2026-09-01") and len(gold) == 99
ok &= np.allclose(gold.values, raw[gold.index].values) and list(raw.index.difference(gold.index)) == [pd.Timestamp("2026-10-01")]
check(f"gold: the EVDS series month by month, {gold.index[0]:%Y-%m} to {gold.index[-1]:%Y-%m}, October 2026 dropped", ok)

# 9. Version 1's smoothers on monthly data: a trailing 7- and 30-month mean and an EWMA with alpha = 2/31
x = gold.values
ma7 = gold.rolling(7).mean()
ewm = gold.ewm(span=30, adjust=False).mean()
e = [x[0]]
for v in x[1:]:
    e.append(e[-1] + 2 / 31 * (v - e[-1]))
ok = abs(ma7.iloc[6] - x[:7].mean()) < 1e-9 and abs(ma7.iloc[-1] - x[-7:].mean()) < 1e-9 and ma7.iloc[:6].isna().all()
ok &= np.allclose(ewm.values, e)
check("7-month moving average and exponential smoothing (span 30) equal the formulas written out", ok)

# 10. AIC with fixed differences is the AIC of the ARMA model of the differenced series (same data, same likelihood)
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

tr, _ = split(co2_v2())
a1 = SARIMAX(tr, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12), simple_differencing=True).fit(disp=False).aic
w = tr.diff().diff(12).dropna()
a2 = ARIMA(w.values, order=(1, 0, 1), seasonal_order=(0, 0, 1, 12), trend="n").fit().aic
a3 = fit_sarima(tr, (1, 1, 1), (0, 1, 1, 12)).aic
check("AIC of SARIMA(1,1,1)(0,1,1)_12 = AIC of ARMA on the differenced series (state-space form within 0.1)",
      abs(a1 - a2) < 0.05 and abs(a3 - a1) < 0.1)

print(f"\n{failed} check(s) failed.")
sys.exit(1 if failed else 0)
