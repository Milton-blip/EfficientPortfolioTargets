#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradesList — per-account rebalance toward averaged target sleeves,
respecting:
  - No asset or cash transfers across accounts
  - Illiquid AUTOMATTIC in "WING Trust" (never trade)
  - Minimize realized gains (never sell more shares than owned; only sell inside same account)
  - Use only assets in the portfolio; if a sleeve is absent in an account, allow BUY via sleeve proxy
  - Use account-specific identifiers when possible

Outputs:
  1) CSV of trades (with AverageCost and CapGain_$)
  2) holdings_aftertrades_<yyyy-mm-dd>.csv
  3) PDF report (per-account tables + tax-status summary)

Run:
  python TradesList11.py --vol 0.08
"""

import os, re, math, argparse, datetime
from pathlib import Path
import io, requests

import numpy as np
import pandas as pd

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fontTools.ttLib import TTFont

# =========================
# ----- Configuration -----
# =========================
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

EST_TAX_RATE = {
    "HSA": 0.00,
    "ROTH IRA": 0.00,
    "Trust": 0.20,
    "Taxable": 0.15,
}

def is_automattic(sym: str, name: str) -> bool:
    s = str(sym).strip().upper()
    n = str(name).strip().upper()
    return ("AUTOMATTIC" in n) or (s == "AUTOMATTIC")

def is_wing_trust(acct: str) -> bool:
    a = str(acct).strip().upper()
    return ("WING" in a) and ("TRUST" in a)

# =========================
# ---- Utility Helpers ----
# =========================
def today_str() -> str:
    return datetime.date.today().isoformat()

def script_basename() -> str:
    return Path(__file__).stem

def fmt_currency(x: float) -> str:
    if x is None or (isinstance(x,float) and (math.isnan(x) or math.isinf(x))):
        return "$0.00"
    neg = x < 0
    s = f"${abs(x):,.2f}"
    return f"({s})" if neg else s

def fmt_number(x: float) -> str:
    try:
        return f"{x:,.0f}"
    except Exception:
        return "0"

def right_cell(pdf: FPDF, w: float, h: float, text: str, border=0):
    pdf.cell(w, h, text, border=border, align="R")

def assign_tax_status(acct: str) -> str:
    if not isinstance(acct, str): return DEFAULT_TAX_STATUS
    low = acct.lower()
    for pat, status in ACCOUNT_TAX_STATUS_RULES:
        if re.search(pat, low):
            return status
    return DEFAULT_TAX_STATUS

def match_col(df: pd.DataFrame, candidates):
    cols = [c for c in df.columns]
    norm = {c.lower().replace(" ",""): c for c in cols}
    for key in candidates:
        k = key.lower().replace(" ","")
        for nc, orig in norm.items():
            if k in nc:
                return orig
    return None

def map_sleeve(sym, name):
    s = str(sym).upper().strip()
    n = str(name).upper().strip()
    if is_automattic(s, n):
        return "Illiquid_Automattic"
    if s in MAP_TO_SLEEVE:
        return MAP_TO_SLEEVE[s]
    if "INFLATION" in n:
        return "TIPS"
    if any(k in n for k in ["UST","TREAS","STRIP"]):
        return "Treasuries"
    return "US_Core"

def is_cashlike(sym: str) -> bool:
    return str(sym).upper() in {"SPAXX","VMFXX","FDRXX","BIL","CASH"}

# =========================
# --- Unicode font setup ---
# =========================
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
            r = requests.get(url, timeout=30, headers={"User-Agent":"python-requests"})
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
    raise RuntimeError(f"Could not fetch Unicode TTF. Last error: {last_err}")

# =========================
# -------- Main -----------
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol", type=float, default=0.08, help="Target volatility tag for filenames/titles (default 0.08).")
    args = parser.parse_args()
    vol_pct_tag = int(round(args.vol * 100))

    # ---------- Load target weights ----------
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
    W_avg.name = "TargetWeight"

    # ---------- Load holdings ----------
    if not Path("holdings.csv").exists():
        raise SystemExit("holdings.csv not found.")

    h = pd.read_csv("holdings.csv")
    h.columns = [c.strip() for c in h.columns]

    col_price = match_col(h, ["Price","Current Price","Last Price","CurrentPrice"])
    col_qty   = match_col(h, ["Quantity","Shares"])
    col_val   = match_col(h, ["Value","Market Value","Current Value"])
    col_acct  = match_col(h, ["Account"])
    col_sym   = match_col(h, ["Symbol","Ticker","Identifier"])
    col_name  = match_col(h, ["Name","Description","Security Name"])
    col_avgc  = match_col(h, ["AverageCost","Average Cost","AvgCost","Avg Cost","Cost Basis","CostBasis"])

    if not col_price:
        raise SystemExit("Price column not found in holdings.csv")

    def to_num(x):
        s = pd.Series(x).astype(str)
        s = s.replace({r"[\$,]": "", r"\(": "-", r"\)": ""}, regex=True)
        s = s.replace({"nan":"0","None":"0"}, regex=False)
        return pd.to_numeric(s, errors="coerce")

    h["Price"] = to_num(h[col_price]).fillna(0.0)
    h["Quantity"] = to_num(h[col_qty]).fillna(0.0) if col_qty else 0.0
    if col_val:
        h["Value"] = to_num(h[col_val]).fillna(h["Quantity"]*h["Price"])
    else:
        h["Value"] = h["Quantity"]*h["Price"]

    h["Account"] = h[col_acct] if col_acct else "Default"
    h["Symbol"]  = h[col_sym]  if col_sym  else ""
    h["Name"]    = h[col_name] if col_name else ""
    if col_avgc:
        h["AverageCost"] = to_num(h[col_avgc]).fillna(0.0)
    else:
        h["AverageCost"] = 0.0

    # sleeves & tax status
    h["Sleeve"] = [map_sleeve(sy, nm) for sy, nm in zip(h["Symbol"], h["Name"])]
    h["TaxStatus"] = [assign_tax_status(a) for a in h["Account"]]

    # account-aware ident selection
    h["_ident"] = h["Symbol"].astype(str).fillna("")
    by_acct_sleeve_ident = h.groupby(["Account", "Sleeve", "_ident"], as_index=False)["Value"].sum()
    canon_acct = {}
    for (acct, sleeve), df_s in by_acct_sleeve_ident.groupby(["Account","Sleeve"]):
        ident = df_s.sort_values("Value", ascending=False)["_ident"].iloc[0]
        canon_acct[(acct, sleeve)] = ident

    # global fallback per sleeve
    by_sleeve_ident = h.groupby(["Sleeve", "_ident"], as_index=False)["Value"].sum()
    canon_global = {}
    for s in W_avg.index:
        df_s = by_sleeve_ident[by_sleeve_ident["Sleeve"] == s]
        if not df_s.empty:
            ident = df_s.sort_values("Value", ascending=False)["_ident"].iloc[0]
            canon_global[s] = ident
        else:
            canon_global[s] = FALLBACK_PROXY.get(s)

    # maps for fast lookup
    price_map = h.groupby("_ident")["Price"].median().to_dict()

    # shares and cost per account-ident
    # weighted avg cost per (Account,_ident)
    def _wavg(g):
        q = g["Quantity"].astype(float)
        ac = g["AverageCost"].astype(float)
        tot = q.sum()
        return 0.0 if tot == 0 else float((ac*q).sum()/tot)
    avgc_map = (
        h.groupby(["Account","_ident"], as_index=False)
         .apply(lambda g: pd.Series({"AvgCostWA": _wavg(g)}))
    )
    avgc_dict = {(r["Account"], r["_ident"]): float(r["AvgCostWA"]) for _, r in avgc_map.iterrows()}

    qty_dict = h.groupby(["Account","_ident"])["Quantity"].sum().to_dict()

    # ---------- Build per-account sleeve deltas ----------
    acct_tot = h.groupby("Account")["Value"].sum()
    rows = []

    def round_shares(delta_dollars, price, ident, for_sell=False, max_sell_shares=0.0):
        if price <= 0:
            return 0.0
        raw = delta_dollars / price
        if for_sell:
            raw = max(-max_sell_shares, raw)  # cap sells by available shares
        if is_cashlike(ident):
            return round(raw, 2)
        return round(raw, 1)

    for acct, dfA in h.groupby("Account"):
        A_total = float(acct_tot.get(acct, 0.0))
        if A_total <= 0:
            continue

        cur_val = dfA.groupby("Sleeve")["Value"].sum()

        # freeze Automattic value inside this account
        illq_val = 0.0
        has_illq_mask = [is_automattic(sy, nm) for sy, nm in zip(dfA["Symbol"], dfA["Name"])]
        if any(has_illq_mask):
            illq_val = dfA.loc[has_illq_mask, "Value"].sum()

        investable_total = max(0.0, A_total - illq_val)

        W_inv = W_avg.copy()
        if "Illiquid_Automattic" in W_inv.index:
            W_inv = W_inv.drop(index="Illiquid_Automattic")
        W_inv = W_inv / W_inv.sum() if W_inv.sum() > 0 else W_inv

        tgt_val = (W_inv * investable_total)
        all_sleeves = sorted(set(cur_val.index).union(set(tgt_val.index)))
        tgt_val = tgt_val.reindex(all_sleeves).fillna(0.0)
        cur_val = cur_val.reindex(all_sleeves).fillna(0.0)

        sleeve_delta = tgt_val - cur_val

        # Build raw trades; constrain sells by shares in this account
        acct_trades = []
        for sleeve, delta in sleeve_delta.items():
            if sleeve == "Illiquid_Automattic":
                continue

            ident_owned = canon_acct.get((acct, sleeve), None)
            ident = ident_owned if ident_owned is not None else canon_global.get(sleeve, None)
            if ident is None:
                continue

            price = float(price_map.get(ident, 0.0))
            if price <= 0 and sleeve in FALLBACK_PROXY:
                proxy = FALLBACK_PROXY[sleeve]
                price = float(price_map.get(proxy, 0.0))
                if price > 0:
                    ident = proxy  # OK to buy proxy if absent
            if price <= 0:
                continue

            if abs(delta) < 5.0:
                continue

            # SELL only if we actually own it in this account; BUY allowed even if we don't
            if delta < 0:
                if ident_owned is None:
                    # No holdings in this account for this sleeve/ident → don't SELL
                    continue
                owned_sh = float(qty_dict.get((acct, ident_owned), 0.0))
                if owned_sh <= 0:
                    continue
                shares = round_shares(delta, price, ident_owned, for_sell=True, max_sell_shares=owned_sh)
                if shares == 0.0:
                    continue
                wavg_cost = float(avgc_dict.get((acct, ident_owned), 0.0))
                capgain = (price - wavg_cost) * abs(shares)
                acct_trades.append({
                    "Account": acct,
                    "TaxStatus": assign_tax_status(acct),
                    "Identifier": ident_owned,
                    "Sleeve": sleeve,
                    "Action": "SELL",
                    "Shares_Delta": shares,   # negative
                    "Price": price,
                    "AverageCost": wavg_cost,
                    "Delta_$": round(shares * price, 2),  # negative $
                    "CapGain_$": round(capgain, 2),
                })
                # update qty_dict to prevent over-selling in subsequent sleeves (same ident)
                qty_dict[(acct, ident_owned)] = owned_sh + shares  # shares is negative
            else:
                # BUY (ident could be owned ident or sleeve proxy if absent)
                shares = round_shares(delta, price, ident, for_sell=False)
                if shares == 0.0:
                    continue
                # BUY has no realized gains
                wavg_cost = float(avgc_dict.get((acct, ident), 0.0))
                acct_trades.append({
                    "Account": acct,
                    "TaxStatus": assign_tax_status(acct),
                    "Identifier": ident,
                    "Sleeve": sleeve,
                    "Action": "BUY",
                    "Shares_Delta": shares,   # positive
                    "Price": price,
                    "AverageCost": wavg_cost,
                    "Delta_$": round(shares * price, 2),  # positive $
                    "CapGain_$": 0.0,
                })

        # Try to keep account roughly cash-neutral by pushing small residuals into Cash
        if acct_trades:
            dfT = pd.DataFrame(acct_trades)
            net_flow = float(dfT["Delta_$"].sum())  # +BUY, -SELL
            if abs(net_flow) > 0.01:
                # Adjust Cash if available; otherwise leave warning
                cash_ident = None
                # Prefer owned cash in account
                if ("Cash" in W_inv.index) or any(dfA["Symbol"].str.upper().isin(["SPAXX","VMFXX","FDRXX","BIL"])):
                    # choose account-specific cash if present; else generic BIL
                    for cash_sym in ["SPAXX","VMFXX","FDRXX","BIL"]:
                        if ((acct, cash_sym) in qty_dict) or (cash_sym in price_map):
                            cash_ident = cash_sym
                            break
                if cash_ident and cash_ident in price_map and price_map[cash_ident] > 0:
                    p = float(price_map[cash_ident])
                    # If net_flow > 0, we need to SELL cash to fund other buys
                    # If net_flow < 0, we need to BUY cash (consume residual sells)
                    action = "SELL" if net_flow > 0 else "BUY"
                    shares = round(net_flow / p, 2)  # allow cents in cash
                    if action == "BUY":
                        # net_flow negative; BUY cash is positive shares
                        shares = abs(shares)
                        delta_d = shares * p
                    else:
                        # action SELL; shares should be negative
                        shares = -abs(shares)
                        delta_d = shares * p
                        # cap sells of cash to available shares
                        owned_cash = float(qty_dict.get((acct, cash_ident), 0.0))
                        if shares < -owned_cash:
                            shares = -owned_cash
                            delta_d = shares * p
                    if abs(shares) > 0:
                        acct_trades.append({
                            "Account": acct,
                            "TaxStatus": assign_tax_status(acct),
                            "Identifier": cash_ident,
                            "Sleeve": "Cash",
                            "Action": "BUY" if shares > 0 else "SELL",
                            "Shares_Delta": shares,
                            "Price": p,
                            "AverageCost": float(avgc_dict.get((acct, cash_ident), 0.0)),
                            "Delta_$": round(delta_d, 2),
                            "CapGain_$": 0.0
                        })
                else:
                    print(f"[WARN] Residual cash flow in '{acct}': {fmt_currency(net_flow)}")

        rows.extend(acct_trades)

    tx = pd.DataFrame(rows)

    # Ensure column order & presence
    desired_cols = [
        "Account","TaxStatus","Identifier","Sleeve","Action",
        "Shares_Delta","Price","AverageCost","Delta_$","CapGain_$"
    ]
    for c in desired_cols:
        if c not in tx.columns:
            tx[c] = "" if c in {"Account","TaxStatus","Identifier","Sleeve","Action"} else 0.0
    tx = tx[desired_cols]

    # ---------- Write CSV ----------
    base = script_basename()
    date = today_str()
    csv_out = f"{base}_Trades_{date}.csv"
    tx.to_csv(csv_out, index=False)
    print(f"CSV written: {os.path.abspath(csv_out)}")

    # ---------- Build post-trade holdings snapshot ----------
    after = h.copy()
    after["_key"] = after["Account"].astype(str) + "||" + after["_ident"].astype(str)

    if not tx.empty:
        # net share deltas (apply by shares!)
        share_deltas = (
            tx.assign(_key=tx["Account"] + "||" + tx["Identifier"])
              .groupby("_key", as_index=True)["Shares_Delta"]
              .sum().to_dict()
        )

        # Add rows for buys of identifiers not yet in holdings for that account
        have_keys = set(after["_key"].astype(str))
        need_keys = set(share_deltas.keys()) - have_keys
        if need_keys:
            inv_canon_global = {v: k for k, v in canon_global.items() if v is not None}
            add_rows = []
            for k in need_keys:
                acct, ident = k.split("||", 1)
                sleeve_guess = inv_canon_global.get(ident, None)
                if sleeve_guess is None:
                    inv_proxy = {v: k for k, v in FALLBACK_PROXY.items()}
                    sleeve_guess = inv_proxy.get(ident, "US_Core")
                row = {
                    "Account": acct,
                    "TaxStatus": assign_tax_status(acct),
                    "Name": ident,
                    "Symbol": ident,
                    "Sleeve": sleeve_guess,
                    "_ident": ident,
                    "Quantity": 0.0,
                    "Price": float(price_map.get(ident, 0.0)),
                    "AverageCost": 0.0,
                    "Value": 0.0,
                }
                add_rows.append(row)
            if add_rows:
                after = pd.concat([after, pd.DataFrame(add_rows)], ignore_index=True)

        def apply_delta(group: pd.DataFrame) -> pd.DataFrame:
            acct, ident = group.name  # (Account, _ident)
            k = f"{acct}||{ident}"
            g = group.copy()
            shares = float(share_deltas.get(k, 0.0))
            if shares != 0.0:
                g.loc[:, "Quantity"] = g["Quantity"] + shares
                g.loc[:, "Value"] = g["Quantity"] * g["Price"]
            return g

        after = after.groupby(["Account","_ident"], group_keys=False).apply(apply_delta, include_groups=False)
        after = after[after["Quantity"].abs() > 1e-6].copy()
        after["Value"] = after["Quantity"] * after["Price"]

    holdings_after_out = f"holdings_aftertrades_{date}.csv"
    preferred = ["Account","TaxStatus","Name","Symbol","Sleeve","Quantity","Price","AverageCost","Value"]
    cols = [c for c in preferred if c in after.columns] + [c for c in after.columns if c not in preferred and not c.startswith("_")]
    after[cols].to_csv(holdings_after_out, index=False)
    print(f"Holdings-after written: {os.path.abspath(holdings_after_out)}")

    # ---------- Summaries ----------
    if tx.empty:
        print("No trades generated; skipping PDF.")
        return

    tx["Buy_$"]  = np.where(tx["Action"]=="BUY",  tx["Delta_$"], 0.0)
    tx["Sell_$"] = np.where(tx["Action"]=="SELL", -tx["Delta_$"], 0.0)

    # Per-account summary
    acc_sum = (
        tx.groupby(["Account","TaxStatus"], as_index=False)
          .agg(**{
              "Total_Buys":   ("Buy_$","sum"),
              "Total_Sells":  ("Sell_$","sum"),
              "Net_CapGain":  ("CapGain_$","sum"),
          })
    )
    acc_sum["Est_Tax"] = acc_sum.apply(lambda r: EST_TAX_RATE.get(r["TaxStatus"], 0.15) * r["Net_CapGain"], axis=1)

    # Tax-status summary
    by_status = (
        tx.groupby("TaxStatus", as_index=False)
          .agg(**{
              "Total_Buys":   ("Buy_$","sum"),
              "Total_Sells":  ("Sell_$","sum"),
              "Net_CapGain":  ("CapGain_$","sum"),
          })
    )
    by_status["Est_Tax"] = by_status.apply(lambda r: EST_TAX_RATE.get(r["TaxStatus"], 0.15) * r["Net_CapGain"], axis=1)

    # ---------- PDF ----------
    ensure_unicode_font()
    pdf = FPDF(orientation="P", unit="mm", format="Letter")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.add_font("Unicode", "", str(FONT_PATH))
    pdf.set_font("Unicode", size=12)

    title = f"Transaction List - Target {vol_pct_tag}% Vol (Real Terms)"
    pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Columns fit within ~178mm
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
        pdf.set_font("Unicode", size=10)
        for (hdr, w, al) in col_defs:
            pdf.cell(w, 7, hdr, border=1, align=al)
        pdf.ln(7)

    def draw_row(row):
        pdf.set_font("Unicode", size=9)
        vals = [
            row["Identifier"],
            row["Sleeve"],
            row["Action"],
            fmt_number(row["Shares_Delta"]),
            fmt_currency(row["Price"]),
            fmt_currency(row["AverageCost"]),
            fmt_currency(row["Delta_$"]),
            fmt_currency(row["CapGain_$"]),
        ]
        for (hdr, w, al), v in zip(col_defs, vals):
            if al == "R":
                right_cell(pdf, w, 7, str(v), border=1)
            else:
                pdf.cell(w, 7, str(v), border=1, align=al)
        pdf.ln(7)

    def draw_kv(label, value):
        pdf.set_font("Unicode", size=10)
        label_w = 65
        val_w   = 40
        pdf.cell(label_w, 6, label, border=0, align="L")
        right_cell(pdf, val_w, 6, value, border=0)
        pdf.ln(6)

    # Per-account sections
    for (acct, tax), g in tx.sort_values(["Account","Action","Sleeve","Identifier"]).groupby(["Account","TaxStatus"]):
        pdf.ln(2)
        pdf.set_font("Unicode", size=11)
        pdf.cell(0, 7, f"Account: {acct}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Unicode", size=10)
        pdf.cell(0, 6, f"Tax Status: {tax}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        draw_header_row()
        for _, r in g.iterrows():
            draw_row(r)

        srow = acc_sum[(acc_sum["Account"]==acct) & (acc_sum["TaxStatus"]==tax)].iloc[0]
        pdf.ln(2)
        draw_kv("Total Buys",                fmt_currency(srow["Total_Buys"]))
        draw_kv("Total Sells",               fmt_currency(srow["Total_Sells"]))
        draw_kv("Net Realized Capital Gain", fmt_currency(srow["Net_CapGain"]))
        draw_kv("Est Cap Gains Tax",         fmt_currency(srow["Est_Tax"]))
        pdf.ln(2)

    # Tax status summary
    pdf.ln(4)
    pdf.set_font("Unicode", size=11)
    pdf.cell(0, 7, "Tax Status Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    ts_cols = [("Tax Status", 40, "L"),
               ("Total Buys", 35, "R"),
               ("Total Sells",35, "R"),
               ("Net CapGain",35, "R"),
               ("Est Tax",    35, "R")]
    for hdr, w, al in ts_cols:
        pdf.cell(w, 7, hdr, border=1, align=al)
    pdf.ln(7)

    pdf.set_font("Unicode", size=9)
    for _, r in by_status.iterrows():
        vals = [
            r["TaxStatus"],
            fmt_currency(r["Total_Buys"]),
            fmt_currency(r["Total_Sells"]),
            fmt_currency(r["Net_CapGain"]),
            fmt_currency(r["Est_Tax"]),
        ]
        for (hdr, w, al), v in zip(ts_cols, vals):
            if al == "R":
                right_cell(pdf, w, 7, v, border=1)
            else:
                pdf.cell(w, 7, str(v), border=1, align=al)
        pdf.ln(7)

    pdf_out = f"{script_basename()}_{vol_pct_tag}vol_{date}.pdf"
    pdf.output(pdf_out)
    print(f"PDF written: {os.path.abspath(pdf_out)}")


if __name__ == "__main__":
    main()