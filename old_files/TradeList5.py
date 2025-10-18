#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradeList6.py
- Per-account sleeve-based rebalance, but trades planned at the security level.
- SELLs are capped by available shares per (Account, Symbol). Never oversells.
- Automattic in WING Trust is illiquid (no trades). Other WING holdings can trade.
- Outputs: CSV trades, holdings_aftertrades_<date>.csv, PDF with summaries.
- --vol (default 0.08) only tags filenames/titles (weights come from your scenario CSVs).
"""

import os, re, math, argparse, datetime
from pathlib import Path
import numpy as np
import pandas as pd

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ---------------- Configuration ----------------
SCENARIOS = ["Base","Disinflation","Reflation","HardLanding","Stagflation","Geopolitical"]

MAP_TO_SLEEVE = {
    'IVW':'US_Growth','VOOG':'US_Growth','AMZN':'US_Growth',
    'SCHB':'US_Core','DFAU':'US_Core','SCHM':'US_Core',
    'SCHA':'US_SmallValue','VBR':'US_SmallValue',
    'IUSV':'US_Value','VTV':'US_Value','VOOV':'US_Value','MGV':'US_Value',
    'VXUS':'Intl_DM','VPL':'Intl_DM','FNDF':'Intl_DM','FNDC':'Intl_DM',
    'VWO':'EM','EMXC':'EM','FNDE':'EM','TSM':'EM',
    'XLE':'Energy','VDE':'Energy',
    'AGG':'IG_Core','SCHZ':'IG_Core',
    'VWOB':'EM_USD','BNDX':'IG_Intl_Hedged',
    'SPAXX':'Cash','FDRXX':'Cash','VMFXX':'Cash','BIL':'Cash'
}

FALLBACK_PROXY = {
    "US_Core":"SCHB","US_Value":"VTV","US_SmallValue":"VBR","US_Growth":"IVW",
    "Intl_DM":"VXUS","EM":"VWO","Energy":"XLE",
    "IG_Core":"AGG","Treasuries":"IEF","TIPS":"TIP",
    "EM_USD":"VWOB","IG_Intl_Hedged":"BNDX","Cash":"BIL"
}

ACCOUNT_TAX_STATUS_RULES = [
    (r"\broth\b|\broth ira\b|vanguard roth|schwab roth", "ROTH IRA"),
    (r"\bhsa\b|fidelity hsa",                            "HSA"),
    (r"\bwing\b.*\btrust\b|\btrust\b",                   "Trust"),
]
DEFAULT_TAX_STATUS = "Taxable"

EST_TAX_RATE = { "HSA":0.00, "ROTH IRA":0.00, "Trust":0.20, "Taxable":0.15 }

# ---------------- Helpers ----------------
def today_str(): return datetime.date.today().isoformat()
def script_basename(): return Path(__file__).stem

def is_automattic(sym: str, name: str) -> bool:
    s = str(sym).strip().upper()
    n = str(name).strip().upper()
    return ("AUTOMATTIC" in n) or (s == "AUTOMATTIC")

def is_wing_trust(acct: str) -> bool:
    a = str(acct).strip().upper()
    return ("WING" in a) and ("TRUST" in a)

def assign_tax_status(acct: str) -> str:
    if not isinstance(acct, str): return DEFAULT_TAX_STATUS
    low = acct.lower()
    for pat, status in ACCOUNT_TAX_STATUS_RULES:
        if re.search(pat, low): return status
    return DEFAULT_TAX_STATUS

def match_col(df: pd.DataFrame, candidates):
    lut = {c.lower().replace(" ",""): c for c in df.columns}
    for key in candidates:
        k = key.lower().replace(" ","")
        for nc, orig in lut.items():
            if k in nc: return orig
    return None

def map_sleeve(sym, name):
    s = str(sym).upper().strip()
    n = str(name).upper().strip()
    if is_automattic(s, n): return "Illiquid_Automattic"
    if s in MAP_TO_SLEEVE:  return MAP_TO_SLEEVE[s]
    if "INFLATION" in n:    return "TIPS"
    if any(k in n for k in ["UST","TREAS","STRIP"]): return "Treasuries"
    return "US_Core"

def fmt_currency(x: float) -> str:
    try:
        if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))): x = 0.0
        neg = x < 0
        s = f"${abs(x):,.2f}"
        return f"({s})" if neg else s
    except Exception:
        return "$0.00"

def fmt_number(x: float) -> str:
    try: return f"{x:,.0f}"
    except Exception: return "0"

def right_cell(pdf: FPDF, w: float, h: float, text: str, border=0):
    pdf.cell(w, h, text, border=border, align="R")

def is_cashlike(sym: str) -> bool:
    return str(sym).upper() in {"SPAXX","VMFXX","FDRXX","BIL","CASH"}

def round_shares_from_dollars(delta_dollars, price, cash=False):
    if price <= 0: return 0.0
    q = delta_dollars/price
    return round(q, 2 if cash else 1)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", type=float, default=0.08, help="Target vol tag (default 0.08) for filenames/titles.")
    args = ap.parse_args()
    vol_pct_tag = int(round(args.vol * 100))

    # ---- Load W_avg across scenarios ----
    alloc_files = [f"allocation_targetVol_{vol_pct_tag}_{s}_Real.csv" for s in SCENARIOS]
    missing = [f for f in alloc_files if not Path(f).exists()]
    if missing:
        raise SystemExit("Missing allocation files:\n  " + "\n  ".join(missing))

    W_list = []
    for f in alloc_files:
        s = pd.read_csv(f, index_col=0).squeeze("columns").astype(float)
        W_list.append(s)
    W_avg = pd.concat(W_list, axis=1).mean(axis=1).clip(lower=0)
    W_avg = W_avg / W_avg.sum()
    if "Illiquid_Automattic" in W_avg.index:
        pass  # fine; we’ll strip it from investable per-account

    # ---- Load holdings ----
    if not Path("holdings.csv").exists():
        raise SystemExit("holdings.csv not found.")
    h = pd.read_csv("holdings.csv")
    h.columns = [c.strip() for c in h.columns]

    col_price = match_col(h, ["Price","Current Price","Last Price","CurrentPrice"])
    col_qty   = match_col(h, ["Quantity","Shares"])
    col_val   = match_col(h, ["Value","Market Value"])
    col_acct  = match_col(h, ["Account"])
    col_sym   = match_col(h, ["Symbol","Ticker","Identifier"])
    col_name  = match_col(h, ["Name","Description","Security Name"])
    col_avgc  = match_col(h, ["AverageCost","Average Cost","AvgCost","Avg Cost"])

    if not col_price or not col_acct or not col_sym:
        raise SystemExit("holdings.csv must include Account, Symbol, and Price/CurrentPrice columns.")

    def to_num(s):
        if isinstance(s, (pd.Series, list)): s = pd.Series(s)
        s = s.astype(str).replace({r"[\$,]": "", r"\(": "-", r"\)": ""}, regex=True)
        s = s.replace({"nan":"0","None":"0"})
        return pd.to_numeric(s, errors="coerce")

    h["Price"]    = to_num(h[col_price]).fillna(0.0)
    h["Quantity"] = to_num(h[col_qty]).fillna(0.0) if col_qty else 0.0
    h["Value"]    = to_num(h[col_val]).fillna(h["Quantity"]*h["Price"]) if col_val else h["Quantity"]*h["Price"]
    h["Account"]  = h[col_acct].astype(str)
    h["Symbol"]   = h[col_sym].astype(str)
    h["Name"]     = h[col_name].astype(str) if col_name else h["Symbol"]
    h["AverageCost"] = to_num(h[col_avgc]).fillna(0.0) if col_avgc else 0.0

    h["Sleeve"] = [map_sleeve(sy, nm) for sy, nm in zip(h["Symbol"], h["Name"])]
    h["TaxStatus"] = [assign_tax_status(a) for a in h["Account"]]
    h["_ident"] = h["Symbol"].astype(str)

    # ---- Account sleeve → preferred ident (largest position in that account for that sleeve) ----
    by_acct_sleeve_ident = h.groupby(["Account","Sleeve","_ident"])["Value"].sum().reset_index()
    canon_acct = {}
    for (acct, sleeve), df in by_acct_sleeve_ident.groupby(["Account","Sleeve"]):
        canon_acct[(acct, sleeve)] = df.sort_values("Value", ascending=False)["_ident"].iloc[0]

    # Global fallback per sleeve
    by_sleeve_ident = h.groupby(["Sleeve","_ident"])["Value"].sum().reset_index()
    canon_global = {}
    for s, df in by_sleeve_ident.groupby("Sleeve"):
        canon_global[s] = df.sort_values("Value", ascending=False)["_ident"].iloc[0] if not df.empty else FALLBACK_PROXY.get(s)

    # Price lookup
    price_map = h.groupby("_ident")["Price"].median().to_dict()

    # ---- Build trades per account, capping sells by available shares ----
    acct_tot = h.groupby("Account")["Value"].sum()
    rows = []

    for acct, dfA in h.groupby("Account", sort=False):
        A_total = float(acct_tot.get(acct, 0.0))
        if A_total <= 0: continue

        # Illiquid Automattic value in this account (no trades)
        mask_illq = [is_automattic(sy, nm) for sy, nm in zip(dfA["Symbol"], dfA["Name"])]
        illq_val  = float(dfA.loc[mask_illq, "Value"].sum()) if any(mask_illq) else 0.0

        investable_total = max(0.0, A_total - illq_val)

        W_inv = W_avg.copy()
        if "Illiquid_Automattic" in W_inv.index:
            W_inv = W_inv.drop(index="Illiquid_Automattic")
        W_inv = W_inv / W_inv.sum() if W_inv.sum() > 0 else W_inv

        # Current sleeve values *in this account*
        cur_val = dfA.groupby("Sleeve")["Value"].sum()
        sleeves = sorted(set(cur_val.index).union(set(W_inv.index)))
        cur_val = cur_val.reindex(sleeves).fillna(0.0)
        tgt_val = (W_inv * investable_total).reindex(sleeves).fillna(0.0)

        sleeve_delta = tgt_val - cur_val  # dollars to buy(+) / sell(-) for each sleeve in this account

        # Precompute positions per sleeve in this account
        # Each entry is a list of dicts with ident, qty, price, avgc, value
        sleeve_positions = {}
        for s, g in dfA.groupby("Sleeve", sort=False):
            poss = []
            for _, r in g.iterrows():
                poss.append({
                    "ident": str(r["_ident"]),
                    "qty":   float(r["Quantity"]),
                    "price": float(r["Price"]),
                    "avgc":  float(r["AverageCost"]),
                    "name":  str(r["Name"]),
                })
            sleeve_positions[s] = poss

        for sleeve, d_dollars in sleeve_delta.items():
            if sleeve == "Illiquid_Automattic":
                continue  # no trades
            d_dollars = float(d_dollars)

            # SELL flow: allocate only across *existing* positions of that sleeve in this account, capped by qty
            if d_dollars < -5.0:
                need = abs(d_dollars)
                poss = sleeve_positions.get(sleeve, [])
                # sort by largest market value first
                poss = sorted(poss, key=lambda p: p["qty"]*p["price"], reverse=True)
                for p in poss:
                    if need <= 0: break
                    ident, qty, price, avgc = p["ident"], p["qty"], p["price"], p["avgc"]
                    if price <= 0 or qty <= 0: continue
                    cashlike = is_cashlike(ident)
                    # Dollar we try to sell on this line
                    try_shares = round_shares_from_dollars(need, price, cash=cashlike)
                    shares_to_sell = min(qty, try_shares)  # CAP by available shares
                    if shares_to_sell <= 0: continue
                    proceeds = shares_to_sell * price
                    capgain  = (price - avgc) * shares_to_sell
                    rows.append({
                        "Account": acct,
                        "TaxStatus": assign_tax_status(acct),
                        "Identifier": ident,
                        "Sleeve": sleeve,
                        "Action": "SELL",
                        "Shares_Delta": -shares_to_sell,
                        "Price": price,
                        "AverageCost": avgc,
                        "Delta_$": -proceeds,          # negative dollar flow (proceeds reduce position)
                        "CapGain_$": capgain
                    })
                    need = max(0.0, need - proceeds)

            # BUY flow: choose account-preferred ident for this sleeve; if none, use global fallback
            elif d_dollars > 5.0:
                ident = canon_acct.get((acct, sleeve), None)
                if ident is None:
                    ident = canon_global.get(sleeve, None)
                if ident is None:  # still nothing
                    continue
                price = float(price_map.get(ident, 0.0))
                if price <= 0:
                    continue
                cashlike = is_cashlike(ident)
                shares = round_shares_from_dollars(d_dollars, price, cash=cashlike)
                if shares <= 0: continue

                # Average cost for buys: use existing AC if position exists in this account; else 0 (new lot)
                rows_ident = dfA[dfA["_ident"] == ident]
                if not rows_ident.empty and rows_ident["Quantity"].sum() > 0:
                    tot_sh = float(rows_ident["Quantity"].sum())
                    wavg_cost = float((rows_ident["AverageCost"] * rows_ident["Quantity"]).sum() / tot_sh) if tot_sh else 0.0
                else:
                    wavg_cost = 0.0

                rows.append({
                    "Account": acct,
                    "TaxStatus": assign_tax_status(acct),
                    "Identifier": ident,
                    "Sleeve": sleeve,
                    "Action": "BUY",
                    "Shares_Delta": shares,
                    "Price": price,
                    "AverageCost": wavg_cost,
                    "Delta_$": shares * price,   # positive cash needed
                    "CapGain_$": 0.0
                })

    tx = pd.DataFrame(rows, columns=[
        "Account","TaxStatus","Identifier","Sleeve","Action",
        "Shares_Delta","Price","AverageCost","Delta_$","CapGain_$"
    ])

    # ---- CSV trades ----
    base = script_basename()
    date = today_str()
    csv_out = f"{base}_Trades_{date}.csv"
    tx.to_csv(csv_out, index=False)
    print(f"CSV written: {os.path.abspath(csv_out)}")

    # ---- Apply trades to build holdings_after (shares-based apply; cannot go negative) ----
    after = h.copy()
    after_key = after["Account"].astype(str) + "||" + after["_ident"].astype(str)

    # net share deltas per (Account, Identifier)
    share_deltas = {}
    if not tx.empty:
        for _, r in tx.iterrows():
            k = f"{r['Account']}||{r['Identifier']}"
            share_deltas[k] = share_deltas.get(k, 0.0) + float(r["Shares_Delta"])

        have_keys = set(after_key)
        need_keys = set(share_deltas.keys()) - have_keys
        if need_keys:
            # Add missing rows (for buys into new tickers in that account)
            inv_proxy = {v:k for k,v in FALLBACK_PROXY.items()}
            add_rows = []
            for k in need_keys:
                acct, ident = k.split("||",1)
                # guess sleeve via reverse maps or default
                sleeve_guess = None
                for (aa, ss), iid in canon_acct.items():
                    if aa == acct and iid == ident: sleeve_guess = ss; break
                if sleeve_guess is None:
                    inv_global = {v:k for k,v in canon_global.items() if v is not None}
                    sleeve_guess = inv_global.get(ident, inv_proxy.get(ident, "US_Core"))
                price = float(price_map.get(ident, 0.0))
                add_rows.append({
                    "Account": acct, "TaxStatus": assign_tax_status(acct),
                    "Name": ident, "Symbol": ident, "Sleeve": sleeve_guess,
                    "_ident": ident, "Quantity": 0.0, "Price": price,
                    "AverageCost": 0.0, "Value": 0.0
                })
            after = pd.concat([after, pd.DataFrame(add_rows)], ignore_index=True)

        # Apply deltas row-wise with non-negativity guard
        def apply_delta(group: pd.DataFrame) -> pd.DataFrame:
            acct = group["Account"].iloc[0]; ident = group["_ident"].iloc[0]
            k = f"{acct}||{ident}"
            dq = float(share_deltas.get(k, 0.0))
            if abs(dq) < 1e-12: return group
            g = group.copy()
            # apply entirely to the largest-lot row (simpler and consistent with planning)
            idx = (g["Quantity"]*g["Price"]).astype(float).idxmax()
            new_q = float(g.loc[idx, "Quantity"]) + dq
            # cap at zero (should not go negative due to SELL capping, but guard anyway)
            new_q = max(0.0, new_q)
            g.loc[idx, "Quantity"] = new_q
            g["Value"] = g["Quantity"] * g["Price"]
            return g

        after = after.groupby(["Account","_ident"], group_keys=False).apply(apply_delta)

        # Drop ~zero positions
        after = after[after["Quantity"].abs() > 1e-9].copy()
        after["Value"] = after["Quantity"] * after["Price"]

    holdings_after_out = f"holdings_aftertrades_{date}.csv"
    preferred = ["Account","TaxStatus","Name","Symbol","Sleeve","Quantity","Price","AverageCost","Value"]
    cols = [c for c in preferred if c in after.columns] + [c for c in after.columns if c not in preferred and not c.startswith("_")]
    after[cols].to_csv(holdings_after_out, index=False)
    print(f"Holdings-after written: {os.path.abspath(holdings_after_out)}")

    # ---- Per-account cash-flow sanity (WING Trust should ~net to zero unless cash exists) ----
    if not tx.empty:
        flow = tx.groupby("Account")["Delta_$"].sum().sort_values()
        for acct, amt in flow.items():
            if abs(amt) > 1.0 and (is_wing_trust(acct) or True):
                # warn for all accounts; user can check cash availability
                print(f"[WARN] Residual cash flow in '{acct}': {fmt_currency(amt)}")

    # ---- Summaries ----
    if tx.empty:
        print("No trades generated; skipping PDF.")
        return

    tx["Buy_$"]  = np.where(tx["Action"]=="BUY",  tx["Delta_$"], 0.0)
    tx["Sell_$"] = np.where(tx["Action"]=="SELL", -tx["Delta_$"], 0.0)  # positive sell proceeds

    acc_sum = tx.groupby(["Account","TaxStatus"], as_index=False).agg(
        Total_Buys   = ("Buy_$","sum"),
        Total_Sells  = ("Sell_$","sum"),
        Net_CapGain  = ("CapGain_$","sum"),
    )
    acc_sum["Est_Tax"] = acc_sum.apply(lambda r: EST_TAX_RATE.get(r["TaxStatus"], 0.15) * r["Net_CapGain"], axis=1)

    by_status = tx.groupby("TaxStatus", as_index=False).agg(
        Total_Buys   = ("Buy_$","sum"),
        Total_Sells  = ("Sell_$","sum"),
        Net_CapGain  = ("CapGain_$","sum"),
    )
    by_status["Est_Tax"] = by_status.apply(lambda r: EST_TAX_RATE.get(r["TaxStatus"], 0.15) * r["Net_CapGain"], axis=1)

    # ---- PDF (Helvetica; ASCII expected) ----
    pdf = FPDF(orientation="P", unit="mm", format="Letter")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    title = f"Transaction List — Target {vol_pct_tag}% Vol (Real Terms)"
    pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Table columns (fit portrait width ~178mm)
    col_defs = [
        ("Identifier",    32, "L"),
        ("Sleeve",        24, "L"),
        ("Action",        14, "C"),
        ("Shares_Delta",  20, "R"),
        ("Price",         20, "R"),
        ("AverageCost",   20, "R"),
        ("Delta_$",       24, "R"),
        ("CapGain_$",     24, "R"),
    ]
    col_widths  = [c[1] for c in col_defs]
    col_aligns  = [c[2] for c in col_defs]

    def draw_header_row():
        pdf.set_font("Helvetica", size=10)
        for (hdr, w, al) in col_defs:
            pdf.cell(w, 7, hdr, border=1, align=al)
        pdf.ln(7)

    def draw_row(row):
        pdf.set_font("Helvetica", size=9)
        vals = [
            row["Identifier"], row["Sleeve"], row["Action"],
            fmt_number(row["Shares_Delta"]),
            fmt_currency(row["Price"]),
            fmt_currency(row["AverageCost"]),
            fmt_currency(row["Delta_$"]),
            fmt_currency(row["CapGain_$"]),
        ]
        for v, (hdr, w, al) in zip(vals, col_defs):
            if al == "R": right_cell(pdf, w, 7, str(v), border=1)
            else:         pdf.cell(w, 7, str(v), border=1, align=al)
        pdf.ln(7)

    def draw_kv(label, value):
        pdf.set_font("Helvetica", size=10)
        label_w = 65; val_w = 40
        pdf.cell(label_w, 6, label, border=0, align="L")
        right_cell(pdf, val_w, 6, value, border=0)
        pdf.ln(6)

    for (acct, tax), g in tx.sort_values(["Account","Action","Sleeve","Identifier"]).groupby(["Account","TaxStatus"]):
        pdf.ln(2)
        pdf.set_font("Helvetica", size=11)
        pdf.cell(0, 7, f"Account: {acct}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 6, f"Tax Status: {tax}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        draw_header_row()
        for _, r in g.iterrows(): draw_row(r)

        srow = acc_sum[(acc_sum["Account"]==acct) & (acc_sum["TaxStatus"]==tax)].iloc[0]
        pdf.ln(2)
        draw_kv("Total Buys",                fmt_currency(srow["Total_Buys"]))
        draw_kv("Total Sells",               fmt_currency(srow["Total_Sells"]))
        draw_kv("Net Realized Capital Gain", fmt_currency(srow["Net_CapGain"]))
        draw_kv("Est Cap Gains Tax",         fmt_currency(srow["Est_Tax"]))
        pdf.ln(2)

    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 7, "Tax Status Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    ts_cols = [("Tax Status", 40, "L"),
               ("Total Buys", 35, "R"),
               ("Total Sells",35, "R"),
               ("Net CapGain",35, "R"),
               ("Est Tax",    35, "R")]
    for hdr, w, al in ts_cols: pdf.cell(w, 7, hdr, border=1, align=al)
    pdf.ln(7)

    pdf.set_font("Helvetica", size=9)
    for _, r in by_status.iterrows():
        vals = [
            r["TaxStatus"],
            fmt_currency(r["Total_Buys"]),
            fmt_currency(r["Total_Sells"]),
            fmt_currency(r["Net_CapGain"]),
            fmt_currency(r["Est_Tax"]),
        ]
        for (hdr, w, al), v in zip(ts_cols, vals):
            if al == "R": right_cell(pdf, w, 7, v, border=1)
            else:         pdf.cell(w, 7, str(v), border=1, align=al)
        pdf.ln(7)

    pdf_out = f"{script_basename()}_{vol_pct_tag}vol_{date}.pdf"
    pdf.output(pdf_out)
    print(f"PDF written: {os.path.abspath(pdf_out)}")


if __name__ == "__main__":
    main()