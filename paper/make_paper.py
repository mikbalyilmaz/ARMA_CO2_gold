"""Writes every number and table of the project note from ../figures/*.csv, so that the text cannot drift away
from the analysis. Run from inside paper/:  python make_paper.py

Output: numbers.tex (one macro per quoted number) and tab_accuracy.tex (body of the accuracy table).
"""

import csv
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "figures"
OUT = Path(__file__).resolve().parent


def read(name):
    with open(FIG / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def dec(x, d):
    s = f"{float(x):.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "$-$")


def pval(x, d=3):
    x = float(x)
    return "$<$0.001" if x < 0.001 else f"{x:.{d}f}"


def kpss_p(x):
    x = float(x)
    return "$\\le$0.01" if x <= 0.01 else ("$\\ge$0.10" if x >= 0.0999 else f"{x:.3f}")


m = {}
facts = read("table6_co2_facts")[0]
m["CoFirst"], m["CoLast"] = facts["first"][:7], facts["last"][:7]
m["CoMonths"], m["CoMissing"] = facts["months"], facts["missing_months"]
NUM = {"1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
m["CoMissingWord"] = NUM.get(facts["missing_months"], facts["missing_months"])
m["CoMissingList"] = facts["missing_list"].replace(" ", ", ")
m["CoFirstValue"], m["CoLastValue"] = dec(facts["first_value"], 1), dec(facts["last_value"], 1)
m["GrowthEarly"], m["GrowthLate"] = dec(facts["growth_1965_1974"], 2), dec(facts["growth_1988_1997"], 2)
m["GrowthNinetyEight"] = dec(facts["growth_1998"], 2)
m["TrainMonths"], m["TestMonths"] = facts["train_months"], facts["test_months"]

cyc = {r["month"]: r["seasonal_effect_ppm"] for r in read("table5_seasonal_cycle")}
m["CycleAmp"] = dec(cyc["amplitude"], 1)
m["CycleMay"], m["CycleOct"] = dec(cyc["5"], 2), dec(cyc["10"], 2)

ur = {r["series"]: r for r in read("table1_unit_roots")}
m["AdfLevelP"] = dec(ur["level"]["adf_p"], 2)
m["AdfDDP"] = pval(ur["diff and seasonal diff"]["adf_p"])
m["AdfDiffP"] = pval(ur["diff"]["adf_p"])
m["KpssLevelP"] = kpss_p(ur["level"]["kpss_p"])
m["KpssDiffP"] = kpss_p(ur["diff"]["kpss_p"])
m["KpssDDP"] = kpss_p(ur["diff and seasonal diff"]["kpss_p"])

byd = {r["d"]: r for r in read("table2_v1_aic_by_d")}
for d, key in (("0", "Zero"), ("1", "One"), ("2", "Two"), ("3", "Three")):
    m[f"AicD{key}"] = dec(byd[d]["aic"], 1)
    m[f"OrderD{key}"] = f"({byd[d]['best_p']},{d},{byd[d]['best_q']})"

sel = read("table3_sarima_selection")
m["SelCount"] = str(len(sel))
m["SelAic"], m["SelSecondAic"] = dec(sel[0]["aic"], 2), dec(sel[1]["aic"], 2)
m["SelNonConverged"] = str(sum(r["converged"] == "False" for r in sel))

ch = {r["term"]: r for r in read("table4_chosen_model")}
m["ThetaHat"], m["ThetaHatSe"] = dec(ch["ma.L1"]["estimate"], 3), dec(ch["ma.L1"]["se"], 3)
m["BigThetaHat"], m["BigThetaHatSe"] = dec(ch["ma.S.L12"]["estimate"], 3), dec(ch["ma.S.L12"]["se"], 3)
m["SigmaSq"] = dec(ch["sigma2"]["estimate"], 4)
for L, key in (("12", "Twelve"), ("24", "TwentyFour"), ("36", "ThirtySix")):
    m[f"LbP{key}"] = dec(ch[f"LjungBox_Q{L}"]["se"], 2)

acc = {r["model"]: r for r in read("table8_forecast_accuracy")}
KEY = {"ARIMA(1,1,1), version 1": "ArimaA", "ARIMA(2,1,3), version 1": "ArimaB", "Seasonal naive with drift": "Snaive",
       "Holt-Winters, additive": "Hw", "SARIMA(0,1,1)(0,1,1)_12": "Sarima", "SARIMA, version 1's data": "SarimaVone"}
for name, key in KEY.items():
    r = acc[name]
    m[f"Mae{key}"], m[f"Rmse{key}"], m[f"Bias{key}"] = dec(r["mae"], 2), dec(r["rmse"], 2), dec(r["bias"], 2)
    if r.get("coverage95"):
        m[f"Cover{key}"] = f"{100 * float(r['coverage95']):.0f}"
        m[f"Inside{key}"] = str(round(48 * float(r["coverage95"])))
conv = acc["ARIMA(2,1,3), version 1"]["converged"]
m["ArimaBConv"] = ("its likelihood maximisation did not converge" if conv == "False"
                   else "its likelihood maximisation converged")
smp = read("table8b_sarima_samples")
m["SigmaVtwo"], m["SigmaFilled"], m["SigmaSkipped"] = (dec(r["sigma2"], 3) for r in smp)
m["MaeSkipped"] = dec(smp[2]["mae"], 2)
yr = {r["year"]: r for r in read("table9_sarima_errors_by_year")}
m["ErrNinetyEight"], m["ErrNinetyNine"] = dec(yr["1998"]["mean_error"], 2), dec(yr["1999"]["mean_error"], 2)
m["ErrTwoThousandOne"] = dec(yr["2001"]["mean_error"], 2)

# Table: forecast accuracy
SHOWN = {"ARIMA(1,1,1), version 1": "ARIMA(1,1,1), v1", "ARIMA(2,1,3), version 1": "ARIMA(2,1,3), v1",
         "Seasonal naive with drift": "Seasonal naive + drift", "Holt-Winters, additive": "Holt--Winters",
         "SARIMA(0,1,1)(0,1,1)_12": "SARIMA $(0,1,1)(0,1,1)_{12}$", "SARIMA, version 1's data": "same, v1 data"}
lines = []
for name, lab in SHOWN.items():
    r = acc[name]
    cov = f"{100 * float(r['coverage95']):.0f}" if r.get("coverage95") else "--"
    lines.append(f"{lab} & {dec(r['mae'], 2)} & {dec(r['rmse'], 2)} & {dec(r['bias'], 2)} & {cov} \\\\")
(OUT / "tab_accuracy.tex").write_text("\n".join(lines) + "\n\\bottomrule\n", encoding="utf-8")

# Gold
gpath = FIG / "table12_gold_facts.csv"
if gpath.exists():
    g = read("table12_gold_facts")[0]
    m["GoFirst"], m["GoLast"], m["GoMonths"], m["GoReturns"] = g["first"], g["last"], g["months"], g["returns"]
    m["GoPriceFirst"], m["GoPriceLast"] = f"{float(g['price_first']):,.0f}", f"{float(g['price_last']):,.0f}"
    m["GoPriceMin"], m["GoPriceMinMonth"] = f"{float(g['price_min']):,.0f}", g["price_min_month"]
    m["GoMean"], m["GoSd"] = dec(g["mean_r"], 2), dec(g["sd_r"], 2)
    m["GoSkew"], m["GoKurt"], m["GoJbP"] = dec(g["skew_r"], 2), dec(g["exkurt_r"], 2), dec(g["jb_p"], 2)
    m["GoBestP"], m["GoBestQ"] = g["aic_best_p"], g["aic_best_q"]
    m["GoBicP"], m["GoBicQ"] = g["bic_best_p"], g["bic_best_q"]
    m["GoAicBest"], m["GoAicZero"] = dec(g["aic_best"], 1), dec(g["aic_00"], 1)
    m["GoLbP"], m["GoLbSqP"] = pval(g["lb_r_p_12"]), pval(g["lb_r2_p_12"])
    m["GoAcOne"] = dec(read("table13_gold_acf")[0]["acf"], 2)
    m["GoOosFirst"], m["GoOosLast"], m["GoOosMonths"] = g["oos_first"], g["oos_last"], g["oos_months"]
    m["GoOosNotMean"] = g["oos_months_not_mean"]
    m["GoRmseMean"], m["GoRmseZero"], m["GoRmseArma"] = (dec(g[k], 2) for k in ("rmse_mean", "rmse_zero", "rmse_arma"))
    m["GoRtwoArma"], m["GoRtwoZero"] = dec(g["r2os_arma"], 3), dec(g["r2os_zero"], 3)
    gu = {r["series"]: r for r in read("table10_gold_unit_roots")}
    m["GoAdfLevelP"], m["GoAdfRetP"] = dec(gu["log price"]["adf_p"], 2), pval(gu["monthly log return"]["adf_p"])
    m["GoKpssLevelP"] = kpss_p(gu["log price"]["kpss_p"])
    m["GoLbSqSixP"] = dec(g["lb_r2_p_6"], 3)
    m["GoRmseArmaRaw"] = dec(g["rmse_arma"], 3)

(OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()), encoding="utf-8")
print(f"Wrote numbers.tex ({len(m)} macros) and tab_accuracy.tex.")
