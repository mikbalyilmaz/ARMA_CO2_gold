"""Writes data/gold_evds_monthly.csv from the EVDS export data/EVDS_13-10-2026.xlsx. Run from the repository root:

    python data/prepare_gold.py

Series TP.ALTINPIYASA.KAP03 of the CBRT data system EVDS: "Altın - Kapanış Fiyatı - USD/ons", gold closing price on
Borsa İstanbul in US dollars per troy ounce, monthly, observation value "Bitiş" (the last closing price of the month).
The export was made on 13 October 2026; its last row (October 2026) is the price of a month that had not ended and
is dropped.

Author: Muhammed Ikbal Yilmaz
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "EVDS_13-10-2026.xlsx"
DOWNLOADED = "2026-10-13"


def read_evds(path=SRC):
    raw = pd.read_excel(path, header=None)
    if str(raw.iloc[0, 0]).strip() != "Tarih" or str(raw.iloc[0, 1]).strip() != "TP_ALTINPIYASA_KAP03":
        raise ValueError("unexpected header in the EVDS export")
    rows = raw.iloc[1:, :2].dropna()
    rows = rows[rows[0].astype(str).str.fullmatch(r"\d{4}-\d{2}")]
    s = pd.Series(pd.to_numeric(rows[1]).values, index=pd.to_datetime(rows[0] + "-01"), name="gold_usd_oz")
    return s.sort_index().asfreq("MS")


if __name__ == "__main__":
    s = read_evds()
    incomplete = pd.Timestamp(DOWNLOADED).to_period("M").to_timestamp()
    dropped = s[s.index >= incomplete]
    s = s[s.index < incomplete]
    s.index.name = "month"
    s.to_csv(HERE / "gold_evds_monthly.csv", float_format="%.4f")
    meta = {"source": "CBRT EVDS, series TP.ALTINPIYASA.KAP03 (Altın - Kapanış Fiyatı - USD/ons, data source BİST)",
            "frequency": "monthly", "observation": "Bitiş (last closing price of the month)", "unit": "USD per troy ounce",
            "downloaded": DOWNLOADED, "first": f"{s.index[0]:%Y-%m}", "last": f"{s.index[-1]:%Y-%m}",
            "months": int(len(s)), "dropped_incomplete_month": [f"{d:%Y-%m}" for d in dropped.index]}
    (HERE / "gold_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
