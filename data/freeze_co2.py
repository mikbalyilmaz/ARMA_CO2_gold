"""Writes the CO2 series shipped with statsmodels to data/co2_mauna_loa_weekly.csv, so that the analyses read a
fixed file. Run from the repository root:  python data/freeze_co2.py

Source (statsmodels.datasets.co2): Keeling, C. D. and T. P. Whorf (2004). Atmospheric CO2 concentrations derived from
flask air samples at sites in the SIO network. In Trends: A Compendium of Data on Global Change. Carbon Dioxide
Information Analysis Center, Oak Ridge National Laboratory. Weekly values, Mauna Loa, March 1958 to December 2001,
in parts per million (ppm); weeks without a sample are missing. Public domain.

Author: Muhammed Ikbal Yilmaz
"""

from pathlib import Path

import statsmodels.api as sm

out = Path(__file__).resolve().parent / "co2_mauna_loa_weekly.csv"
d = sm.datasets.co2.load_pandas().data
d.index.name = "date"
d.to_csv(out, float_format="%.4f")
print(f"wrote {out.name}: {len(d)} weeks, {int(d['co2'].isna().sum())} without a sample")
