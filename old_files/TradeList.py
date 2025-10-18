#!/usr/bin/env python3
"""
Rebalance holdings to aggregate 8% volatility (real terms) efficient frontier target.
Outputs both a CSV trade file and a PDF report.

Usage:
    /usr/local/bin/python3 rebalance_to_target.py
"""

import os, re
import pandas as pd
import numpy as np
from fpdf import FPDF
from datetime import date

# --- Unicode font bootstrap (put near the top, after imports) ---
from pathlib import Path
import io, requests
from fontTools.ttLib import TTFont, TTLibError


# --- Unicode font bootstrap (robust, with fallbacks) ---
from pathlib import Path
import io, requests
from fontTools.ttLib import TTFont, TTLibError

FONT_DIR  = Path(__file__).parent / "fonts"
FONT_PATH = FONT_DIR / "UnicodeSans.ttf"   # generic name for whatever we fetch

# Try multiple reliable TTF sources (first one usually works)
FONT_URLS = [
    # Google Noto Sans (primary)
    "https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans-Regular.ttf",
    # DejaVu Sans (alt mirror on main)
    "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
    # Source Sans 3 (fallback)
    "https://github.com/adobe-fonts/source-sans/raw/release/TTF/SourceSans3-Regular.ttf",
]

def _is_valid_ttf_bytes(b: bytes) -> bool:
    if len(b) < 4:
        return False
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
            data = f.read(8)
        if data[:4] not in (b"\x00\x01\x00\x00", b"true", b"typ1", b"OTTO"):
            return False
        TTFont(str(p))  # deep check
        return True
    except Exception:
        return False

def ensure_unicode_font():
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    if _is_valid_ttf_path(FONT_PATH):
        return  # good already

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
            else:
                last_err = f"Invalid TTF bytes from {url}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    raise RuntimeError(f"Could not fetch a valid Unicode TTF. Last error: {last_err}")
# --- end unicode font bootstrap ---




# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
SCENARIOS = ["Base","Disinflation","Reflation","HardLanding","Stagflation","Geopolitical"]
ALLOC_FILES = [f"allocation_targetVol_8_{s}_Real.csv" for s in SCENARIOS]
FALLBACK = {
    "US_Core":"SCHB","US_Value":"VTV","US_SmallValue":"VBR","US_Growth":"IVW",
    "Intl_DM":"VXUS","EM":"VWO","Energy":"XLE",
    "IG_Core":"AGG","Treasuries":"IEF","TIPS":"TIP",
    "EM_USD":"VWOB","IG_Intl_Hedged":"BNDX","Cash":"BIL"
}
PRICE_MIN = 5.0   # ignore trades smaller than this $

# ---------------------------------------------------------------------
# Output filenames (include current date)
# ---------------------------------------------------------------------
today_str = date.today().isoformat()  # e.g. "2025-10-13"
PDF_OUTPUT = f"TransactionList - {today_str}.pdf"
CSV_OUTPUT = f"TransactionList - {today_str}.csv"


# ---------------------------------------------------------------------
# ILLIQUID HOLDINGS CONFIG
# ---------------------------------------------------------------------
# Automattic (private company) is illiquid only when held in these accounts
AUTO_ILLQ_ACCOUNTS = {"WING TRUST"}

# ---------------------------------------------------------------------
# LOAD INPUT FILES
# ---------------------------------------------------------------------
missing = [f for f in ALLOC_FILES if not os.path.exists(f)]
if missing:
    raise SystemExit(f"Missing allocation files:\n  {missing}")

if not os.path.exists("holdings.csv"):
    raise SystemExit("holdings.csv not found.")

# Aggregate target weights (average of all scenarios)
W_list = [pd.read_csv(f, index_col=0).squeeze("columns") for f in ALLOC_FILES]
W_avg = pd.concat(W_list, axis=1).mean(axis=1).clip(lower=0).astype(float)
W_avg /= W_avg.sum()
W_avg.name = "TargetWeight"

# ---------------------------------------------------------------------
# LOAD HOLDINGS
# ---------------------------------------------------------------------
h = pd.read_csv("holdings.csv")
h.columns = [c.strip() for c in h.columns]

# Flexible column mapping
def match_col(possible_names):
    for p in possible_names:
        for c in h.columns:
            if p.lower().replace(" ","") in c.lower().replace(" ",""):
                return c
    return None

col_price = match_col(["Price","Current Price","Last Price"])
col_qty   = match_col(["Quantity","Shares"])
col_val   = match_col(["Value","Market Value"])
col_acct  = match_col(["Account"])
col_sym   = match_col(["Symbol","Ticker"])
col_name  = match_col(["Name","Description"])

if not col_price:
    raise SystemExit("Price column not found in holdings.csv")

h["Price"] = pd.to_numeric(h[col_price].replace({r"[\$,]":""}, regex=True), errors="coerce").fillna(0.0)
h["Quantity"] = pd.to_numeric(h[col_qty], errors="coerce").fillna(0.0) if col_qty else 0.0
h["Value"] = pd.to_numeric(h[col_val], errors="coerce").fillna(h["Quantity"]*h["Price"]) if col_val else h["Quantity"]*h["Price"]
h["Account"] = h[col_acct] if col_acct else "Default"
h["Symbol"] = h[col_sym] if col_sym else ""
h["Name"] = h[col_name] if col_name else ""

# ---------------------------------------------------------------------
# SLEEVE MAPPING
# ---------------------------------------------------------------------
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

def is_automattic(symbol: str, name: str) -> bool:
    s = str(symbol).upper()
    n = str(name).upper()
    return ("AUTOMATTIC" in n) or (s in {"AUTOMATTIC"})

# ---- Illiquid detection: Automattic cannot be traded (only in WING Trust) ----
def is_automattic(symbol: str, name: str, account: str) -> bool:
    s = str(symbol or "").upper()
    n = str(name or "").upper()
    a = str(account or "").upper().strip()
    if a not in AUTO_ILLQ_ACCOUNTS:
        return False
    return ("AUTOMATTIC" in n) or (s == "AUTOMATTIC")

def map_sleeve(sym, name, account):
    s = str(sym).upper(); n = str(name).upper()
    # Only flag as illiquid if it's Automattic in a designated account (e.g., WING Trust)
    if is_automattic(sym, name, account):
        return "Illiquid_Automattic"
    if s in MAP_TO_SLEEVE: return MAP_TO_SLEEVE[s]
    if "INFLATION" in n: return "TIPS"
    if any(k in n for k in ["UST","TREAS","STRIP"]): return "Treasuries"
    return "US_Core"

h["Sleeve"] = [map_sleeve(sym, nm, acct) for sym, nm, acct in zip(h["Symbol"], h["Name"], h["Account"])]



# ---------------------------------------------------------------------
# DETERMINE CANONICAL IDENTIFIER FOR EACH SLEEVE
# ---------------------------------------------------------------------
h["_ident"] = h["Symbol"].astype(str)
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

# ---------------------------------------------------------------------
# REBALANCE TO TARGET
# ---------------------------------------------------------------------
acct_tot = h.groupby("Account")["Value"].sum()
rows = []

def is_cashlike(sym): return sym.upper() in {"SPAXX","VMFXX","FDRXX","BIL"}

def round_shares(delta_dollars, price, ident):
    if price <= 0: return 0
    if is_cashlike(ident): return round(delta_dollars/price, 2)
    return round(delta_dollars/price, 1)

ILLQ = "Illiquid_Automattic"  # same sleeve name you already use

for acct, dfA in h.groupby("Account"):
    A_total = float(dfA["Value"].sum())
    if A_total <= 0:
        continue

    # Current sleeve dollars in this account
    cur_val = dfA.groupby("Sleeve")["Value"].sum()
    illq_val = float(cur_val.get(ILLQ, 0.0))
    illq_w   = illq_val / A_total if A_total > 0 else 0.0

    # --- Build a per-account target weight vector that locks Automattic ---
    w_base = W_avg.copy()

    # Strip illiquid from the investable set and renormalize to (1 - illq_w) in this account
    w_inv = w_base.drop(index=[ILLQ], errors="ignore")
    inv_sum = float(w_inv.sum())
    if inv_sum > 0:
        w_inv = w_inv / inv_sum * (1.0 - illq_w)
    else:
        w_inv = pd.Series(dtype=float)

    # Reassemble full per-account target weights
    w_tgt_acct = w_inv.copy()
    if illq_w > 0:
        w_tgt_acct.loc[ILLQ] = illq_w

    # Convert to target dollars and align with current sleeves
    tgt_val = (w_tgt_acct * A_total)
    sleeves_all = sorted(set(tgt_val.index).union(cur_val.index))
    tgt_val = tgt_val.reindex(sleeves_all).fillna(0.0)
    cur_val = cur_val.reindex(sleeves_all).fillna(0.0)

    # Raw deltas
    deltas = tgt_val - cur_val

    # Never trade Automattic
    if ILLQ in deltas.index:
        deltas.loc[ILLQ] = 0.0

    # --- Enforce zero external cash flow for this account ---
    has_cash = "Cash" in deltas.index
    if has_cash:
        resid = float(deltas.sum())
        deltas.loc["Cash"] -= resid
    else:
        buy_sum  = float(deltas[deltas > 0].sum())
        sell_sum = float(-deltas[deltas < 0].sum())
        if buy_sum > 0 and sell_sum >= 0:
            scale = min(1.0, sell_sum / buy_sum) if sell_sum > 0 else 0.0
            deltas.loc[deltas > 0] = deltas[deltas > 0] * scale

    # --- Build trades ---
    for sleeve, delta in deltas.items():
        if sleeve == ILLQ:
            continue  # never trade Automattic
        delta = float(delta)
        if abs(delta) < PRICE_MIN:
            continue

        ident = canon.get(sleeve)
        price = float(price_map.get(ident, 0.0))
        if not ident or price <= 0:
            continue

        action = "BUY" if delta > 0 else "SELL"
        shares = round_shares(delta, price, ident)
        if shares == 0:
            continue

        rows.append({
            "Account": acct,
            "Identifier": ident,
            "Sleeve": sleeve,
            "Action": action,
            "Shares_Delta": shares,
            "Price": price,
            "Delta_$": round(delta, 2)
        })

tx = pd.DataFrame(rows)
tx.to_csv(CSV_OUTPUT, index=False)

# ---------------------------------------------------------------------
# GENERATE PDF SUMMARY
# ---------------------------------------------------------------------
from fpdf import FPDF
from fpdf.enums import XPos, YPos

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
ensure_unicode_font()
pdf.add_font("UnicodeSans", "", str(FONT_PATH))
pdf.set_font("UnicodeSans", size=11)
pdf.cell(
    0, 10,
    "Transaction List – 8% Frontier (Real) — Automattic in WING Trust Held (No Trades)",
    new_x=XPos.LMARGIN, new_y=YPos.NEXT
)

pdf.set_font("UnicodeSans", size=10)
pdf.cell(0, 8, f"Total Transactions: {len(tx)}",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(4)

def cellrow(vals, widths):
    for v, w in zip(vals, widths):
        pdf.cell(w, 8, str(v), border=1)
    pdf.ln(8)

cols = ["Account","Identifier","Sleeve","Action","Shares_Delta","Price","Delta_$"]
widths = [25,30,25,20,25,20,25]

cellrow(cols, widths)
pdf.set_font("UnicodeSans", size=9)
for _, r in tx.iterrows():
    vals = [r[c] for c in cols]
    cellrow(vals, widths)

pdf.output(PDF_OUTPUT)
print(f"\n✅ Rebalance complete.")
print(f"CSV written: {os.path.abspath(CSV_OUTPUT)}")
print(f"PDF written: {os.path.abspath(PDF_OUTPUT)}")