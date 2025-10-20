#!/usr/bin/env python3
import argparse, os
import numpy as np, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", required=True)
    ap.add_argument("--out", default="returns/sleeve_returns.csv")
    ap.add_argument("--months", type=int, default=36)
    rng = np.random.default_rng(42)
    args = ap.parse_args()

    h = pd.read_csv(args.holdings)
    colmap = {c.lower(): c for c in h.columns}
    def pick(*cands):
        for c in cands:
            if c.lower() in colmap: return colmap[c.lower()]
        raise SystemExit(f"Missing columns; need one of: {cands}")

    sym_col  = pick("Symbol","Ticker")
    name_col = pick("Name","Security","Description")
    sleeve_col = colmap.get("sleeve")

    # very simple sleeve inference if Sleeve empty
    def infer_sleeve(sym, name):
        s = str(sym).upper()
        n = str(name).upper()
        if "CASH" in n or s in {"BIL","SHV","SGOV"}: return "Cash"
        if "TIPS" in n: return "TIPS"
        if any(k in n for k in ["TREAS","UST","STRIP","BOND 7-10","IEF","TLH","TLT"]): return "Treasuries"
        if any(k in n for k in ["EMERGING","E.M.","EM "]) or "EM " in n or " E M" in n: return "EM"
        if "INTL" in n or "INTERNATIONAL" in n or "EAFE" in n: return "Intl_DM"
        if any(k in n for k in ["VALUE"]) and any(k in n for k in ["US","USA","AMERICA","S&P","RUSSELL","TOTAL"]): return "US_Value"
        if any(k in n for k in ["GROWTH"]) and any(k in n for k in ["US","USA","AMERICA","S&P","RUSSELL","TOTAL"]): return "US_Growth"
        if any(k in n for k in ["SMALL"]) and any(k in n for k in ["VALUE"]): return "US_SmallValue"
        if any(k in n for k in ["AGG","CORE BOND","IG","INVESTMENT GRADE"]) and "INTL" not in n: return "IG_Core"
        if "HEDGED" in n and any(k in n for k in ["INTL","INTERNATIONAL","GLOBAL"]): return "IG_Intl_Hedged"
        if any(k in n for k in ["ENERGY","XLE","VDE","OIL","GAS"]): return "Energy"
        if any(k in n for k in ["AUTOMATTIC","ILLIQ"]): return "Illiquid_Automattic"
        return "US_Core"

    sleeves = []
    for _, r in h.iterrows():
        if sleeve_col and isinstance(r[sleeve_col], str) and r[sleeve_col].strip():
            sleeves.append(str(r[sleeve_col]).strip())
        else:
            sleeves.append(infer_sleeve(r[sym_col], r[name_col]))
    sleeves = sorted(set(sleeves))

    outdir = Path(os.path.dirname(args.out) or ".")
    outdir.mkdir(parents=True, exist_ok=True)

    # simple synthetic return generator (AR(0) noise around sleeve-level drifts)
    # annual drifts/vols -> monthly
    base = {
        "Cash": (0.03, 0.01),
        "US_Core": (0.08, 0.16),
        "US_Value": (0.08, 0.17),
        "US_Growth": (0.09, 0.20),
        "US_SmallValue": (0.09, 0.22),
        "Intl_DM": (0.07, 0.17),
        "EM": (0.09, 0.23),
        "EM_USD": (0.08, 0.18),
        "IG_Core": (0.05, 0.07),
        "IG_Intl_Hedged": (0.045, 0.06),
        "Treasuries": (0.04, 0.09),
        "TIPS": (0.04, 0.08),
        "Energy": (0.07, 0.30),
        "Illiquid_Automattic": (0.0, 0.0),
    }

    months = pd.period_range(end=pd.Timestamp.today(), periods=args.months, freq="M").to_timestamp()
    df = pd.DataFrame({"Date": months})

    for s in sleeves:
        mu_a, vol_a = base.get(s, (0.07, 0.18))
        mu_m = (1+mu_a)**(1/12)-1
        vol_m = vol_a / np.sqrt(12)
        r = rng.normal(mu_m, vol_m, size=args.months)
        df[s] = r

    # Drop sleeves with zero variance (e.g., illiquid) to avoid optimization issues
    keep_cols = ["Date"] + [c for c in df.columns if c == "Date" or df[c].std(skipna=True) > 0]
    df = df[keep_cols]

    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with sleeves: {', '.join([c for c in df.columns if c!='Date'])}")

if __name__ == "__main__":
    main()