#!/usr/bin/env python3
"""
Rebalance holdings to an aggregate target volatility frontier (default 8%, real terms).
Outputs a CSV trade file and a PDF report with per-account summaries and cap gains.

Usage examples:
  .venv312/bin/python TradeList2.py
  .venv312/bin/python TradeList2.py --target_vol 0.10
"""

import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import io, requests
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fontTools.ttLib import TTFont

# ---------------- Runtime-derived names ----------------
BASE_NAME = Path(__file__).stem                 # e.g., "TradeList2"
TODAY = datetime.today().strftime("%Y-%m-%d")   # e.g., "2025-10-13"

# ---------------- Args ----------------
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--target_vol", type=float, default=0.08,
                help="Annualized volatility target as a decimal (default 0.08 for 8%).")
args = ap.parse_args()
TV = int(round(args.target_vol * 100))          # e.g., 8, 10, 12

# ---------------- Unicode font bootstrap (robust) ----------------
FONT_DIR  = Path(__file__).parent / "fonts"
FONT_PATH = FONT_DIR / "UnicodeSans.ttf"
FONT_URLS = [
    "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans-Regular.ttf",
    "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
    "https://github.com/adobe-fonts/source-sans/raw/release/TTF/SourceSans3-Regular.ttf",
]

def _is_valid_ttf_bytes(b: bytes) -> bool:
    if len(b) < 4: return False
    if b[:4] not in (b"\x00\x01\x00\x00", b"true", b"typ1", b"OTTO"):
        return False
    try:
        TTFont(io.BytesIO(b))
        return True
    except Exception:
        return False

def _is_valid_ttf_path(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            hdr = f.read(4)
        if hdr not in (b"\x00\x01\x00\x00", b"true", b"typ1", b"OTTO"):
            return False
        TTFont(str(p))
        return True
    except Exception:
        return False

def ensure_unicode_font():
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    if _is_valid_ttf_path(FONT_PATH):
        return
    last_err = None
    for url in FONT_URLS:
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "python-requests"})
            r.raise_for_status()
            data = r.content
            if _is_valid_ttf_bytes(data):
                with open(FONT_PATH, "wb") as f:
                    f.write(data)
                return
            last_err = f"Invalid TTF bytes from {url}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"Could not fetch a valid Unicode TTF. Last error: {last_err}")
# ---------------- end font bootstrap ----------------


# ---------------- Configuration ----------------
SCENARIOS = ["Base","Disinflation","Reflation","HardLanding","Stagflation","Geopolitical"]
# Load the six scenario target-vol CSVs matching the chosen target_vol (real terms)
ALLOC_FILES = [f"allocation_targetVol_{TV}_{s}_Real.csv" for s in SCENARIOS]

# Canonical fallbacks by sleeve → tradable ticker
FALLBACK = {
    "US_Core":"SCHB","US_Value":"VTV","US_SmallValue":"VBR","US_Growth":"IVW",
    "Intl_DM":"VXUS","EM":"VWO","Energy":"XLE",
    "IG_Core":"AGG","Treasuries":"IEF","TIPS":"TIP",
    "EM_USD":"VWOB","IG_Intl_Hedged":"BNDX","Cash":"BIL"
}
PRICE_MIN = 5.0  # ignore trades smaller than this $

# Outputs named from the script file
PDF_OUTPUT = f"{BASE_NAME} - {TODAY}.pdf"
CSV_OUTPUT = f"{BASE_NAME} - {TODAY}.csv"

# Illiquid private position
ILLQ_SLEEVE = "Illiquid_Automattic"
ILLQ_ACCOUNT = "WING Trust"   # Automattic belongs here; cannot be sold


# ---------------- Helpers ----------------
def is_automattic(symbol: str, name: str) -> bool:
    s = str(symbol).upper()
    n = str(name).upper()
    return ("AUTOMATTIC" in n) or (s == "AUTOMATTIC")

def is_cashlike(sym: str) -> bool:
    return str(sym).upper() in {"SPAXX","VMFXX","FDRXX","BIL"}

def round_shares(delta_dollars: float, price: float, ident: str) -> float:
    if price <= 0:
        return 0.0
    if is_cashlike(ident):
        return round(delta_dollars/price, 2)
    return round(delta_dollars/price, 1)

def fmt_money(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)


# ---------------- Load inputs ----------------
missing = [f for f in ALLOC_FILES if not os.path.exists(f)]
if missing:
    raise SystemExit(f"Missing allocation files for target_vol={args.target_vol:.2%}:\n  " +
                     "\n  ".join(missing))

if not os.path.exists("holdings.csv"):
    raise SystemExit("holdings.csv not found.")

# Aggregate target weights (average of all scenarios)
W_list = [pd.read_csv(f, index_col=0).squeeze("columns") for f in ALLOC_FILES]
W_avg = pd.concat(W_list, axis=1).mean(axis=1).clip(lower=0).astype(float)
W_avg /= W_avg.sum()
W_avg.name = "TargetWeight"

# Load holdings (flexible columns)
h = pd.read_csv("holdings.csv")
h.columns = [c.strip() for c in h.columns]

def match_col(df, possible_names):
    for p in possible_names:
        for c in df.columns:
            if p.lower().replace(" ","") in c.lower().replace(" ",""):
                return c
    return None

col_price = match_col(h, ["Price","Current Price","Last Price"])
col_qty   = match_col(h, ["Quantity","Shares"])
col_val   = match_col(h, ["Value","Market Value"])
col_acct  = match_col(h, ["Account"])
col_sym   = match_col(h, ["Symbol","Ticker"])
col_name  = match_col(h, ["Name","Description"])

# cost basis columns (any of these)
col_avg_cost = match_col(h, ["Avg Cost","Average Cost","Cost Basis/Share","CostBasisPerShare","Cost Basis Per Share"])
col_cost_tot = match_col(h, ["Cost Basis","Total Cost Basis","CostBasis","Cost"])

if not col_price:
    raise SystemExit("Price column not found in holdings.csv")

h["Price"] = pd.to_numeric(h[col_price].replace({r"[\$,]":""}, regex=True), errors="coerce").fillna(0.0)
h["Quantity"] = pd.to_numeric(h[col_qty], errors="coerce").fillna(0.0) if col_qty else 0.0
h["Value"] = (pd.to_numeric(h[col_val], errors="coerce")
              .fillna(h["Quantity"]*h["Price"]) if col_val else h["Quantity"]*h["Price"])
h["Account"] = h[col_acct] if col_acct else "Default"
h["Symbol"] = h[col_sym] if col_sym else ""
h["Name"] = h[col_name] if col_name else ""

# Force Automattic rows to the correct account (safety)
mask_auto = [is_automattic(s, n) for s, n in zip(h["Symbol"], h["Name"])]
if any(mask_auto):
    h.loc[mask_auto, "Account"] = ILLQ_ACCOUNT

# Build per-(Account,Identifier) average cost
h["_ident"] = h["Symbol"].astype(str)

if col_avg_cost:
    h["_avg_cost"] = pd.to_numeric(h[col_avg_cost], errors="coerce")
elif col_cost_tot:
    total_cost = pd.to_numeric(h[col_cost_tot], errors="coerce")
    qty = h["Quantity"].replace(0, np.nan)
    h["_avg_cost"] = (total_cost / qty).replace([np.inf, -np.inf], np.nan)
else:
    h["_avg_cost"] = np.nan

avg_cost_map = (
    h.groupby(["Account","_ident"])["_avg_cost"]
     .median()
     .dropna()
     .to_dict()
)

# ---------------- Sleeve mapping ----------------
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
    'SPAXX':'Cash','FDRXX':'Cash','VMFXX':'Cash'
}
def map_sleeve(sym, name):
    s = str(sym).upper(); n = str(name).upper()
    if is_automattic(sym, name):
        return ILLQ_SLEEVE
    if s in MAP_TO_SLEEVE: return MAP_TO_SLEEVE[s]
    if "INFLATION" in n: return "TIPS"
    if any(k in n for k in ["UST","TREAS","STRIP"]): return "Treasuries"
    return "US_Core"

h["Sleeve"] = [map_sleeve(s, nm) for s, nm in zip(h["Symbol"], h["Name"])]

# Canonical identifier per sleeve (prefer what you already hold)
by_sleeve_ident = h.groupby(["Sleeve","_ident"])["Value"].sum().reset_index()
canon = {}
for s in W_avg.index:
    df_s = by_sleeve_ident[by_sleeve_ident["Sleeve"]==s]
    if not df_s.empty:
        ident = df_s.sort_values("Value", ascending=False)["_ident"].iloc[0]
        canon[s] = ident
    else:
        canon[s] = FALLBACK.get(s, None)

price_map = h.groupby("_ident")["Price"].median().to_dict()
for s, ident in list(canon.items()):
    if ident not in price_map or price_map[ident] <= 0:
        fb = FALLBACK.get(s)
        if fb in price_map and price_map[fb] > 0:
            canon[s] = fb
        else:
            raise SystemExit(f"Missing price for sleeve {s}")

# ---------------- Rebalance to target ----------------
acct_tot = h.groupby("Account")["Value"].sum()
rows = []

for acct, dfA in h.groupby("Account"):
    A_total = float(acct_tot.get(acct, 0.0))
    if A_total <= 0:
        continue

    # Account-level target values by sleeve (pro-rata to this account AUM)
    tgt_val = (W_avg * A_total)

    # Current sleeve totals in this account
    cur_val = dfA.groupby("Sleeve")["Value"].sum()

    # Lock Automattic: target equals current within this account
    if ILLQ_SLEEVE in cur_val.index:
        tgt_val.loc[ILLQ_SLEEVE] = cur_val.loc[ILLQ_SLEEVE]

    # Deltas
    sleeve_delta = (tgt_val - cur_val).reindex(W_avg.index).fillna(tgt_val)

    for sleeve, delta in sleeve_delta.items():
        if sleeve == ILLQ_SLEEVE:
            continue  # never trade Automattic

        delta = float(delta)
        ident = canon.get(sleeve)
        price = float(price_map.get(ident, 0.0))
        if abs(delta) < PRICE_MIN or price <= 0.0 or ident is None:
            continue

        action = "BUY" if delta > 0 else "SELL"
        shares = round_shares(delta, price, ident)

        # Capital gain/loss: only for SELL; use avg-cost if available
        cap_gain = np.nan
        if action == "SELL":
            avg_cost = avg_cost_map.get((acct, ident), np.nan)
            if np.isfinite(avg_cost):
                cap_gain = (price - float(avg_cost)) * abs(shares)
            else:
                cap_gain = np.nan

        rows.append({
            "Account": acct,
            "Identifier": ident,
            "Sleeve": sleeve,
            "Action": action,
            "Shares_Delta": shares,
            "Price": round(price, 4),
            "Delta_$": round(delta, 2),
            "CapGain_$": np.nan if pd.isna(cap_gain) else round(cap_gain, 2),
        })

tx = pd.DataFrame(rows)

# Sort and save CSV
if not tx.empty:
    tx["absd"] = tx["Delta_$"].abs()
    tx = tx.sort_values(["Account","Sleeve","Action","absd"],
                        ascending=[True, True, True, False]).drop(columns="absd")
tx.to_csv(CSV_OUTPUT, index=False)

# ---------------- Per-account summary ----------------
if tx.empty:
    acct_summary = pd.DataFrame(columns=["Account","Total_Buys_$","Total_Sells_$","Realized_CapGain_$"])
else:
    g = tx.copy()
    g["Buy_$"]  = g["Delta_$"].where(g["Delta_$"]>0, 0.0)
    g["Sell_$"] = (-g["Delta_$"]).where(g["Delta_$"]<0, 0.0)
    g["Realized_CapGain_$"] = np.where(
        (g["Action"]=="SELL") & np.isfinite(g["CapGain_$"]),
        g["CapGain_$"], 0.0
    )
    acct_summary = (
        g.groupby("Account")
        .agg({
            "Buy_$": "sum",
            "Sell_$": "sum",
            "Realized_CapGain_$": "sum"
        })
        .rename(columns={
            "Buy_$": "Total_Buys_$",
            "Sell_$": "Total_Sells_$",
            "Realized_CapGain_$": "Realized_CapGain_$"
        })
        .reset_index()
    )
    for c in ["Total_Buys_$","Total_Sells_$","Realized_CapGain_$"]:
        acct_summary[c] = acct_summary[c].round(2)

# ---------------- PDF ----------------
ensure_unicode_font()

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.add_font("UnicodeSans", "", str(FONT_PATH))
pdf.set_font("UnicodeSans", size=12)

title = f"{BASE_NAME} – {TODAY} – Target Vol {args.target_vol:.0%} (Real Terms)"
pdf.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.set_font("UnicodeSans", size=10)
pdf.cell(0, 8, f"Total Transactions: {len(tx)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(2)

# Table header
pdf.set_font("UnicodeSans", size=9)
cols = ["Account","Identifier","Sleeve","Action","Shares_Delta","Price","Delta_$","CapGain_$"]
widths = [32,28,28,18,26,20,24,26]

def row(vals):
    for (v, w, colname) in zip(vals, widths, cols):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if colname in ("Price","Delta_$","CapGain_$"):
                txt = fmt_money(v)
            else:
                txt = f"{v}"
        else:
            txt = str(v)
        pdf.cell(w, 7, txt, border=1)
    pdf.ln(7)

# Header
for c, w in zip(cols, widths):
    pdf.cell(w, 7, c, border=1)
pdf.ln(7)

# Rows
if tx.empty:
    pdf.cell(0, 8, "No trades required.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
else:
    for _, r in tx.iterrows():
        vals = [r[c] for c in cols]
        row(vals)

pdf.ln(6)
pdf.set_font("UnicodeSans", size=11)
pdf.cell(0, 8, "Per-Account Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font("UnicodeSans", size=9)

# Summary table
s_cols = ["Account","Total_Buys_$","Total_Sells_$","Realized_CapGain_$"]
s_w    = [40,35,35,40]
for c, w in zip(s_cols, s_w):
    pdf.cell(w, 7, c, border=1)
pdf.ln(7)

if acct_summary.empty:
    pdf.cell(0, 8, "No activity.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
else:
    for _, r in acct_summary.iterrows():
        pdf.cell(s_w[0], 7, str(r["Account"]), border=1)
        pdf.cell(s_w[1], 7, fmt_money(r["Total_Buys_$"]), border=1)
        pdf.cell(s_w[2], 7, fmt_money(r["Total_Sells_$"]), border=1)
        pdf.cell(s_w[3], 7, fmt_money(r["Realized_CapGain_$"]), border=1)
        pdf.ln(7)

pdf.output(PDF_OUTPUT)

print("✅ Rebalance complete.")
print(f"CSV written: {os.path.abspath(CSV_OUTPUT)}")
print(f"PDF written: {os.path.abspath(PDF_OUTPUT)}")