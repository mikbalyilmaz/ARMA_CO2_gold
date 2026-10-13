"""Box-Jenkins tools: unit-root tests, (seasonal) ARIMA estimation and order selection, benchmark forecasts,
forecast accuracy and residual checks.

Notation: B is the backshift operator, B y_t = y_{t-1}. A seasonal ARIMA(p,d,q)(P,D,Q)_12 model is
    phi(B) Phi(B^12) (1 - B)^d (1 - B^12)^D y_t = theta(B) Theta(B^12) e_t,   e_t ~ WN(0, sigma^2),
with phi, theta of degrees p, q and Phi, Theta of degrees P, Q in B^12. The "airline" model of Box and Jenkins is
(0,1,1)(0,1,1)_12. Models are estimated by exact Gaussian maximum likelihood (Kalman filter, statsmodels).

Author: Muhammed Ikbal Yilmaz
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

Z95 = 1.959963984540054


# ------------------------------------------------------------------------------------------ unit roots
def unit_root(x, trend):
    """ADF (H0: unit root; lag order by AIC, at most 12*(n/100)^(1/4)) and KPSS (H0: stationary) with the same
    deterministic terms: trend = "c" (constant) or "ct" (constant and linear trend)."""
    x = pd.Series(x).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # KPSS warns when its p-value is outside its table (0.01-0.10);
        adf = adfuller(x, regression=trend, autolag="AIC")       # newer statsmodels warn about a future return type
        k = kpss(x, regression=trend, nlags="auto")
    return {"n": len(x), "adf_stat": float(adf[0]), "adf_p": float(adf[1]), "adf_lags": int(adf[2]),
            "kpss_stat": float(k[0]), "kpss_p": float(k[1]), "kpss_lags": int(k[2])}


def transforms(y):
    """Level, first difference, seasonal difference and both, with the deterministic terms of their tests."""
    return {"level": (y, "ct"), "diff": (y.diff(), "c"), "seasonal diff": (y.diff(12), "c"),
            "diff and seasonal diff": (y.diff().diff(12), "c")}


# ------------------------------------------------------------------------------------------ models
def fit_sarima(y, order, seasonal=(0, 0, 0, 0), trend="n"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(y, order=order, seasonal_order=seasonal, trend=trend).fit(disp=False)


def fit_arima_v1(y, order):
    """statsmodels' ARIMA class, as used by version 1 (no constant when d > 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(y, order=order).fit()


def forecast_frame(res, h, index):
    """Mean forecast and 95% interval as a DataFrame on `index`."""
    fc = res.get_forecast(h)
    ci = fc.conf_int(alpha=0.05)
    return pd.DataFrame({"mean": fc.predicted_mean.values, "lo": ci.iloc[:, 0].values, "hi": ci.iloc[:, 1].values},
                        index=index)


def select_sarima(y, p=range(3), q=range(3), P=range(2), Q=range(2), d=1, D=1):
    """All (p,d,q)(P,D,Q)_12 with the given ranges; d and D fixed, so that every AIC is computed on the same
    differenced data and the values are comparable. Returns a DataFrame sorted by AIC."""
    rows = []
    for pp, qq, PP, QQ in itertools.product(p, q, P, Q):
        res = fit_sarima(y, (pp, d, qq), (PP, D, QQ, 12))
        rows.append({"p": pp, "q": qq, "P": PP, "Q": QQ, "aic": res.aic, "bic": res.bic,
                     "converged": bool(res.mle_retvals.get("converged", True))})
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def arima_grid_v1(y, orders=range(4)):
    """Version 1's search: ARIMA(p,d,q), p, d, q in 0..3, statsmodels' ARIMA (no constant when d > 0), the AIC of
    each fit. Returns all fits; fits that fail are recorded with aic = NaN."""
    rows = []
    for p, d, q in itertools.product(orders, orders, orders):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aic = ARIMA(y, order=(p, d, q)).fit().aic
        except Exception:                                   # noqa: BLE001  (as version 1: skip a failing order)
            aic = np.nan
        rows.append({"p": p, "d": d, "q": q, "aic": aic})
    return pd.DataFrame(rows)


def holt_winters(y, h, index):
    """Additive Holt-Winters (level, trend, 12 seasonal states), estimated by least squares (statsmodels)."""
    res = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit()
    return pd.DataFrame({"mean": res.forecast(h).values}, index=index)


def seasonal_naive_drift(y, h, index):
    """Seasonal naive forecast with drift: yhat_{T+k} = y_{T+k-12j} + j g, j = ceil(k/12), where g is the mean of
    y_t - y_{t-12} over the training sample (the average change over twelve months)."""
    y = np.asarray(y, float)
    g = np.mean(y[12:] - y[:-12])                          # average change over twelve months
    out = [y[len(y) - 12 + (k % 12)] + g * (k // 12 + 1) for k in range(h)]
    return pd.DataFrame({"mean": out}, index=index)


# ------------------------------------------------------------------------------------------ evaluation
def accuracy(actual, fc):
    """MAE, RMSE and, when the forecast has an interval, the share of months inside the 95% interval."""
    e = np.asarray(fc["mean"]) - np.asarray(actual)
    out = {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e))}
    if "lo" in fc:
        a = np.asarray(actual)
        out["coverage95"] = float(np.mean((a >= fc["lo"].values) & (a <= fc["hi"].values)))
    return out


def ljung_box(resid, lags, fitted_params=0):
    """Ljung-Box Q at the given lags, with the degrees of freedom reduced by the number of ARMA parameters."""
    r = acorr_ljungbox(pd.Series(resid).dropna(), lags=list(lags), model_df=fitted_params)
    return r.reset_index().rename(columns={"index": "lag"})
