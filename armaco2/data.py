"""Data: monthly CO2 at Mauna Loa and the monthly gold price.

Files (data/):
  co2_mauna_loa_weekly.csv   weekly flask CO2, Mauna Loa, 1958-03 to 2001-12, ppm (from statsmodels; see
                             freeze_co2.py)
  gold_evds_monthly.csv      gold closing price on Borsa Istanbul, US dollars per troy ounce, last close of each
                             month, July 2018 to September 2026 (CBRT EVDS series TP.ALTINPIYASA.KAP03; see
                             prepare_gold.py)

Author: Muhammed Ikbal Yilmaz
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

TRAIN_END = "1997-12-01"     # the split of version 1: training to December 1997,
TEST_START = "1998-01-01"    # 48 test months from January 1998 to December 2001
V2_START = "1965-01-01"      # first month after the last month without any sample (April 1964)


def load_co2_weekly(path=None):
    d = pd.read_csv(path or DATA / "co2_mauna_loa_weekly.csv", parse_dates=["date"], index_col="date")
    return d["co2"]


def co2_monthly(weekly=None):
    """Monthly means of the weekly samples (as in version 1); a month without any sample is missing (NaN)."""
    weekly = load_co2_weekly() if weekly is None else weekly
    return weekly.resample("MS").mean()


def co2_v1(weekly=None):
    """The series of version 1: monthly means, missing months filled with the next available month."""
    y = co2_monthly(weekly)
    return y.fillna(y.bfill())


def co2_v2(weekly=None):
    """The series of version 2: monthly means from January 1965, a complete monthly record with no filled values."""
    y = co2_monthly(weekly)[V2_START:]
    if y.isna().any():
        raise ValueError("missing months after January 1965")
    return y


def split(y):
    return y[:TRAIN_END], y[TEST_START:]


def load_gold(path=None):
    """Monthly gold price (US dollars per troy ounce, last close of the month), indexed by the first day of the month."""
    d = pd.read_csv(path or DATA / "gold_evds_monthly.csv", parse_dates=["month"], index_col="month")
    return d["gold_usd_oz"].asfreq("MS")
