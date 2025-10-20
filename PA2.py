#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cvxpy as cp


# ------------------------
# Sleeve mapping helpers
# ------------------------

MAP_TO_SLEEVE: Dict[str, str] = {
    # Cash-like
    "CASH": "Cash", "BIL": "Cash", "SGOV": "Cash", "SPAXX": "Cash",
    # Treasuries
    "TLT": "Treasuries", "IEF": "Treasuries", "SHY": "Treasuries",
    # TIPS
    "TIP": "TIPS", "SCHP": "TIPS",
    # EM / Intl
    "VWO": "EM", "IEMG": "EM", "EEM": "EM",
    "VEA": "Intl_DM", "IEFA": "Intl_DM", "EFA": "Intl_DM",
    "EMB": "EM_USD",
    # US equities
    "VTI": "US_Core", "ITOT": "US_Core", "SCHB": "US_Core", "VOO": "US_Core", "SPY": "US_Core",
    "VUG": "US_Growth", "IVW": "US_Growth",
    "VTV": "US_Value", "IVE": "US_Value",
    "VBR": "US_SmallValue", "AVUV": "US_SmallValue",
    # Bonds
    "AGG": "IG_Core", "BND": "IG_Core",
    "BNDX": "IG_Intl_Hedged", "IAGG": "IG_Intl_Hedged",
    # Energy proxy
    "XLE": "Energy", "VDE": "Energy",
}

def is_automattic(symbol: str, name: str) -> bool:
    s = str(symbol).upper()
    n = str(name).upper()
    return ("AUTOMATTIC" in n) or (s in {"AUTO", "AUTOM"} and "AUTOMATTIC" in n)

def is_cashlike(symbol: str) -> bool:
    s = str(symbol).upper()
    return s in {"CASH", "BIL", "SGOV", "SPAXX"} or s.startswith("CASH")

def infer_sleeve(symbol: str, name: str) -> str:
    s = str(symbol).upper().strip()
    n = str(name).upper().strip()

    if is_automattic(s, n):
        return "Illiquid_Automattic"
    if s in MAP_TO_SLEEVE:
        return MAP_TO_SLEEVE[s]

    # name keyword fallbacks
    if any(k in n for k in ["UST", "TREAS", "STRIP", "TREASUR"]):
        return "Treasuries"
    if "INFLATION" in n or "TIPS" in n:
        return "TIPS"
    if "ENERGY" in n:
        return "Energy"
    if any(k in n for k in ["EMERGING", " EM", "EM "]):
        return "EM"
    if any(k in n for k in ["DEVELOPED", "INTL", "INTERNATIONAL", "EAFE", "EUROPE", "JAPAN"]):
        return "Intl_DM"
    if "SMALL" in n and "VALUE" in n:
        return "US_SmallValue"
    if "GROWTH" in n:
        return "US_Growth"
    if "VALUE" in n:
        return "US_Value"
    if any(k in n for k in ["CORPORATE", "AGGREGATE"]) and any(k in n for k in ["BOND", "FIXED"]):
        return "IG_Core"
    if "CASH" in n:
        return "Cash"

    return "US_Core"  # safe default


# ------------------------
# IO: holdings & returns
# ------------------------

def _lower_map(columns: List[str]) -> Dict[str, str]:
    return {c.lower(): c for c in columns}

def _has(cols: Dict[str, str], name: str) -> bool:
    return name.lower() in cols

def _col(cols: Dict[str, str], name: str) -> str:
    return cols[name.lower()]

def read_holdings(path: str) -> pd.DataFrame:
    f = Path(path)
    if not f.exists():
        raise SystemExit(f"[ERROR] Holdings file not found: {path}")
    df = pd.read_csv(f)

    cols = _lower_map(df.columns)

    # Required base fields
    for need in ["symbol", "name", "quantity"]:
        if not _has(cols, need):
            raise SystemExit(f"[ERROR] Holdings missing required column: {need!r}")

    sym = _col(cols, "symbol")
    nam = _col(cols, "name")
    qty = _col(cols, "quantity")

    # Optional price/value fields under multiple possible names
    price = None
    for cand in ["pricepershare", "price"]:
        if _has(cols, cand):
            price = _col(cols, cand)
            break

    mv = None
    for cand in ["marketvalue", "market_value", "value", "market value"]:
        if _has(cols, cand):
            mv = _col(cols, cand)
            break

    sleeve_col = _col(cols, "sleeve") if _has(cols, "sleeve") else None

    out = pd.DataFrame({
        "Symbol": df[sym].astype(str),
        "Name": df[nam].astype(str),
        "Quantity": pd.to_numeric(df[qty], errors="coerce").fillna(0.0),
    })

    out["Price"] = pd.to_numeric(df[price], errors="coerce") if price else np.nan
    out["MarketValue"] = pd.to_numeric(df[mv], errors="coerce") if mv else np.nan

    # derive Price from MV/Qty when needed
    need_price = out["Price"].isna() & out["Quantity"].gt(0)
    out.loc[need_price, "Price"] = out.loc[need_price, "MarketValue"] / out.loc[need_price, "Quantity"]

    # derive MV from Qty*Price when needed
    need_mv = out["MarketValue"].isna()
    out.loc[need_mv, "MarketValue"] = out.loc[need_mv, "Quantity"] * out.loc[need_mv, "Price"]

    # ensure sleeve dtype object (prevents pandas warnings on mixed types)
    out["Sleeve"] = pd.Series(index=out.index, dtype="object")
    if sleeve_col is not None:
        raw = df[sleeve_col].astype("string")
        raw = raw.where(~raw.isna(), None)
        out.loc[:, "Sleeve"] = raw

    # fill missing sleeves via inference
    mask_missing = out["Sleeve"].isna() | (out["Sleeve"].astype(str).str.strip() == "")
    if mask_missing.any():
        inferred = [
            infer_sleeve(s, n) if mm else sv
            for s, n, sv, mm in zip(out["Symbol"], out["Name"], out["Sleeve"], mask_missing)
        ]
        out["Sleeve"] = pd.Series(inferred, index=out.index, dtype="object")

    # normalize obvious cash
    cash_mask = out["Symbol"].map(is_cashlike)
    out.loc[cash_mask, "Sleeve"] = "Cash"

    # drop empties
    out = out[(out["Quantity"].abs() > 0) | (out["MarketValue"].abs() > 0)].copy()
    out["MarketValue"] = out["MarketValue"].fillna(0.0)
    out["Price"] = out["Price"].fillna(0.0)

    return out


def read_returns(path: str) -> pd.DataFrame:
    f = Path(path)
    if not f.exists():
        raise SystemExit(
            "[ERROR] Could not find returns file.\n"
            "Place a CSV like:\n"
            "  returns/sleeve_returns.csv\n"
            "with columns: Date,<sleeve1>,<sleeve2>,...\n"
            "Each column should be period returns (e.g., monthly decimal returns)."
        )
    r = pd.read_csv(f)
    if r.empty or r.shape[1] < 2:
        raise SystemExit("[ERROR] Returns file must have a Date column and at least one sleeve column.")
    r = r.dropna(how="all")
    r = r.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    r = r.dropna(axis=1, how="all")
    return r


# ------------------------
# Optimization building blocks
# ------------------------

def normalize_inputs(holdings_sleeve_value: pd.Series, returns_by_sleeve: pd.DataFrame
                     ) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
    hv = holdings_sleeve_value.groupby(level=0).sum()  # index: Sleeve
    hv = hv[hv > 0]
    if hv.empty:
        raise SystemExit("[ERROR] No positive MarketValue sleeves in holdings.")

    hv_w = hv / hv.sum()

    sleeves_h = set(hv_w.index.astype(str))
    sleeves_r = set(returns_by_sleeve.columns.astype(str))
    overlap = sorted(sleeves_h & sleeves_r)

    if not overlap:
        Path("debug_no_overlap.json").write_text(json.dumps({
            "holdings_sleeves": sorted(sleeves_h),
            "returns_sleeves": sorted(sleeves_r),
        }, indent=2))
        raise SystemExit("[WARN] After normalization, no sleeves overlap between holdings and returns.\n"
                         "       Wrote debugging info to debug_no_overlap.json")

    R = returns_by_sleeve[overlap].copy()
    mu = R.mean(axis=0)
    cov = R.cov()
    hv_w = hv_w.reindex(overlap).fillna(0.0)
    return mu, cov, hv_w


def max_return_at_vol(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_vol: float,
    w0: pd.Series | None = None,
    w_min: float = 0.0,
    w_max: float = 1.0,
    l2_to_current: float = 0.0,
) -> pd.Series:
    sleeves = list(mu.index)
    n = len(sleeves)
    if n == 0:
        raise SystemExit("[ERROR] No sleeves available for optimization.")

    w = cp.Variable(n)
    Sigma = cov.reindex(index=sleeves, columns=sleeves).values

    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        w <= w_max,
        cp.quad_form(w, Sigma) <= target_vol ** 2,
    ]

    if w0 is None:
        w0v = np.zeros(n)
    else:
        w0v = w0.reindex(sleeves).fillna(0.0).values

    # concave objective: maximize mu·w - α||w - w0||²
    objective = cp.Maximize(mu.reindex(sleeves).values @ w - l2_to_current * cp.sum_squares(w - w0v))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=30000)
    if prob.status not in {"optimal", "optimal_inaccurate"}:
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=30000)
    if prob.status not in {"optimal", "optimal_inaccurate"}:
        raise SystemExit(f"[ERROR] Solver failed: {prob.status}")

    sol = pd.Series(np.array(w.value).reshape(-1), index=sleeves).clip(lower=0)
    sol = sol / sol.sum() if sol.sum() > 0 else sol
    return sol


# ------------------------
# Reporting helpers
# ------------------------

def print_results(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame) -> None:
    mu_a = mu.reindex(weights.index).fillna(0.0).values
    cov_a = cov.reindex(index=weights.index, columns=weights.index).values
    exp_ret = float(mu_a @ weights.values)
    vol = float(np.sqrt(weights.values @ cov_a @ weights.values))
    sharpe = exp_ret / vol if vol > 0 else np.nan

    print("\nWeights (%):")
    print((weights * 100).round(2).sort_values(ascending=False).to_frame("Weight%"))

    print(f"\nExpected Return %: {(exp_ret*100):.2f}")
    print(f"Volatility %: {(vol*100):.2f}")
    print(f"Sharpe: {sharpe:.2f}")

    mu_sorted = mu.sort_values(ascending=False) * 100
    print("\nMean period returns by sleeve (%), descending:")
    print(mu_sorted.round(3))


# ------------------------
# CLI
# ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Efficient portfolio sleeve optimizer (sleeve targets).")
    p.add_argument("--holdings", required=True, help="Path to holdings CSV.")
    p.add_argument("--returns-file", default="returns/sleeve_returns.csv", help="CSV of sleeve returns (Date,<sleeves...>).")
    p.add_argument("--target-vol", type=float, required=True, help="Target volatility as decimal (e.g. 0.08 for 8%%).")
    p.add_argument("--max-weight", type=float, default=1.0, help="Per-sleeve upper bound (e.g. 0.35).")
    p.add_argument("--min-weight", type=float, default=0.0, help="Per-sleeve lower bound.")
    p.add_argument("--l2-to-current", type=float, default=0.0, help="Penalty toward current sleeve weights (e.g. 0.05).")
    return p.parse_args()


def main():
    args = parse_args()

    holdings = read_holdings(args.holdings)

    illq = holdings.loc[holdings["Sleeve"] == "Illiquid_Automattic", "MarketValue"].sum()
    if illq > 0:
        print(f"[NOTE] Excluding sleeves with no return history from stats: ['Illiquid_Automattic']")

    sleeves_series = holdings.set_index("Sleeve")["MarketValue"]

    R = read_returns(args.returns_file)
    mu, cov, cur_w = normalize_inputs(sleeves_series, R)

    w_opt = max_return_at_vol(
        mu, cov, float(args.target_vol),
        w0=cur_w,
        w_min=float(args.min_weight),
        w_max=float(args.max_weight),
        l2_to_current=float(args.l2_to_current),
    )

    print_results(w_opt, mu, cov)


if __name__ == "__main__":
    main()