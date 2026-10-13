# Changes

## Version 2.0.0

### New

- `armaco2/` holds the data handling, the Box–Jenkins tools (ADF and KPSS tests, seasonal ARIMA estimation, AIC
  selection with the differences fixed, benchmark forecasts, accuracy measures, Ljung–Box) and a runner for the
  CO₂ notebook of version 1.
- CO₂: the series, its twelve-month change and its seasonal cycle; correlograms of the level and its differences;
  seasonal ARIMA $(p,1,q)(P,1,Q)_{12}$ selected by AIC; forecasts for 1998–2001 against the seasonal naive forecast
  with drift and additive Holt–Winters, with 95% intervals and their coverage.
- Gold: the price with version 1's smoothers, unit-root tests, ARMA selection for monthly returns, Ljung–Box tests
  of the returns and of their squares, and one-month-ahead forecasts against the historical mean.
- The data are stored in the repository: the CO₂ series of `statsmodels` (`data/freeze_co2.py`) and the EVDS gold
  export with `data/prepare_gold.py`.
- Unit tests (`tests/run_tests.py`), four figures, CSV tables, and a project note in English whose numbers are
  written from the tables.
- `CITATION.cff`, `requirements.txt`, `.gitignore`, a GitHub Actions workflow that runs the tests.

### Corrections to version 1

| # | Version 1 | Version 2 |
|---|---|---|
| 1 | The order of the CO₂ model was chosen by the smallest AIC over ARIMA$(p,d,q)$, $p,d,q \in \{0,\dots,3\}$. AIC values for different $d$ are likelihoods of different data ($y_t$, $\Delta y_t$, $\Delta^2 y_t$, ...) and cannot be compared. | $d$ and $D$ are set by unit-root tests and the correlogram ($d = D = 1$); AIC then compares 36 seasonal models fitted to the same differenced data. |
| 2 | Both CO₂ models, ARIMA(1,1,1) and the AIC choice ARIMA(2,1,3), have no seasonal part, and `statsmodels` fits no constant when $d = 1$: the forecasts level off (mean absolute error 2.30 and 3.80 ppm over 1998–2001). The likelihood maximisation of ARIMA(2,1,3) does not converge. | The seasonal ARIMA $(0,1,1)(0,1,1)_{12}$ (mean absolute error 0.78 ppm), compared with the seasonal naive forecast with drift and Holt–Winters. |
| 3 | The five months without a CO₂ sample (1958-06, 1958-10, 1964-02 to 1964-04) were filled with the next month. Estimated on these data, the seasonal ARIMA has twice the innovation variance. | The sample starts in January 1965; nothing is filled. |
| 4 | Most cells printed "Elapsed time", computed as `time.time()` without a start time: the seconds since 1970 (about 1.73 × 10⁹). | Removed. |
| 5 | The gold file `gold_london_ons.csv` read by the notebook was not in the repository, so the gold analysis could not be run. | The EVDS export is stored with a script that prepares it. EVDS offers the series from July 2018, so version 1's sample from March 2012 could not be reproduced. |
| 6 | The gold data are monthly, but the moving averages were labelled "7-Day MA" and "30-Day MA"; the price axis said "Price (£)", the histogram axis "Longitude" (from the column name `lon`). | The windows are 7 and 30 months; the price is in US dollars per troy ounce. |
| 7 | A histogram of the price level with a fitted normal density: the price trends, so its histogram describes no distribution. | The distribution of monthly log returns, after unit-root tests on the log price and the returns. |
| 8 | The gold notebook began with Python exercises unrelated to the analysis (variable types, keywords, a function printing "Hello"). | Not carried over; version 1 is kept unchanged in `legacy/`. |
| 9 | The README said "All rights reserved", which contradicts the MIT license of the repository. | The README describes the project; the license is MIT. |

The CO₂ notebook of version 1 ends with the statement that its code "is not my own but was created as a training
portfolio project". Version 2 is written anew; the code of version 1 is run unchanged only to reproduce its numbers.

## Version 1.0.0

First version: ARIMA forecasts of monthly CO₂ at Mauna Loa (`ARMA_CO2_forecast.ipynb`) and moving averages and
exponential smoothing of a monthly gold price (`ons.ipynb`).
