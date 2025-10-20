#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except Exception as e:
    print("cvxpy is required. Install with: pip install cvxpy", file=sys.stderr)
    raise

SLEEVE_MAP = {
    "BIL": "Cash", "SHV": "Cash", "SGOV": "Cash", "CASH": "Cash",
    "SPY": "US_Core", "IVV": "US_Core", "VTI": "US_Core", "ITOT": "US_Core",
    "VOO": "US_Core", "SCHX": "US_Core",
    "IWD": "US_Value", "VTV": "US_Value", "SPYV": "US_Value",
    "IVW": "US_Growth", "VUG": "US_Growth", "SPYG": "US_Growth",
    "IJS": "US_SmallValue", "VBR": "US_SmallValue", "SLYV": "US_SmallValue",
    "EFA": "Intl_DM", "VEA": "Intl_DM", "IEFA": "Intl_DM",
    "EEM": "EM", "VWO": "EM", "IEMG": "EM",
    "EMB": "EM_USD",
    "HYG": "IG_Core", "LQD": "IG_Core", "AGG": "IG_Core", "BND": "IG_Core",
    "BNDX": "IG_Intl_Hedged", "IGIB": "IG_Core",
    "IEF": "Treasuries", "IEI": "Treasuries", "TLT": "Treasuries", "SHY": "Treasuries",
    "TIP": "TIPS", "SCHP": "TIPS",
    "XLE": "Energy", "VDE": "Energy",
}

DEFAULT_RETURNS_REL = "returns/sleeve_returns.csv"
ILLQ = "Illiquid_Automattic"

def parse_args():
    p = argparse.ArgumentParser(
        description="Efficient portfolio target generator."
    )
    p.add_argument(
        "--holdings", required=True,
        help="Path to holdings.csv"
    )
    p.add_argument(
        "--returns-file", default=DEFAULT_RETURNS_REL,
        help="CSV of sleeve returns (Date as first col, then sleeve columns)."
    )
    p.add_argument(
        "--target-vol", type=float, default=None,
        help="Target volatility as a decimal (e.g., 0.08 for 8%%)."
    )
    return p.parse_args()

def read_holdings(path: str) -> pd.DataFrame:
    f = Path(path)
    if not f.exists():
        raise SystemExit(f"[ERROR] Holdings file not found: {path}")
    df = pd.read_csv(f)
    needed_any = {"Symbol", "Name", "Quantity", "Price"}
    if not needed_any.issubset(df.columns):
        raise SystemExit(f"[ERROR] Holdings missing required columns: {sorted(needed_any - set(df.columns))}")
    if "Sleeve" not in df.columns:
        df["Sleeve"] = pd.NA
    df["Sleeve"] = df["Sleeve"].astype("string")
    q = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
    px = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0)
    df["Value"] = q * px
    return df

def infer_sleeves(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Sleeve"] = out["Sleeve"].astype("string").replace("", pd.NA)

    syms = out["Symbol"].astype(str).str.upper().str.strip()
    names = out["Name"].astype(str).str.upper().str.strip()

    inferred = []
    for s, n in zip(syms, names):
        if s in SLEEVE_MAP:
            inferred.append(SLEEVE_MAP[s])
        elif "TREAS" in n or "UST" in n or "STRIP" in n:
            inferred.append("Treasuries")
        elif "INFLATION" in n or "TIPS" in n:
            inferred.append("TIPS")
        elif "ENERGY" in n or "XLE" in n or "VDE" in n:
            inferred.append("Energy")
        else:
            inferred.append("US_Core")

    inferred = pd.Series(inferred, index=out.index, dtype="string")
    mask = out["Sleeve"].isna()
    out.loc[mask, "Sleeve"] = inferred.loc[mask]
    return out

def load_returns(path: str) -> pd.DataFrame:
    f = Path(path)
    if not f.exists():
        raise SystemExit(
            "[ERROR] Could not find returns file. Place a CSV like:\n"
            "  returns/sleeve_returns.csv\n"
            "with columns: Date,<sleeve1>,<sleeve2>,...\n"
            "Each column should be period returns (e.g., monthly decimal returns)."
        )
    r = pd.read_csv(f)
    if "Date" not in r.columns:
        raise SystemExit("[ERROR] returns file must have a 'Date' column")
    r = r.set_index("Date")
    r = r.apply(pd.to_numeric, errors="coerce")
    r = r.dropna(how="all", axis=1).dropna(how="any", axis=0)
    return r

def weights_by_sleeve(df: pd.DataFrame) -> pd.Series:
    by = df.groupby("Sleeve")["Value"].sum()
    total = by.sum()
    if total <= 0:
        raise SystemExit("[ERROR] Total holdings value is 0.")
    w = by / total
    return w

def normalize_inputs(holdings_w: pd.Series, returns_df: pd.DataFrame):
    sleeves_h = set(holdings_w.index)
    sleeves_r = set(returns_df.columns)
    common = sorted(sleeves_h & sleeves_r)
    if ILLQ in common:
        common.remove(ILLQ)
    if len(common) == 0:
        print("[WARN] After normalization, no sleeves overlap between holdings and returns.")
        debug = {
            "holdings_sleeves": sorted(list(sleeves_h)),
            "returns_sleeves": sorted(list(sleeves_r)),
        }
        Path("debug_no_overlap.json").write_text(json.dumps(debug, indent=2))
        return None, None, None
    w = holdings_w.reindex(common).fillna(0.0)
    r = returns_df[common].copy()
    mu = r.mean().values
    cov = r.cov().values
    return w.index.tolist(), mu, cov

def max_return_at_vol(mu: np.ndarray, cov: np.ndarray, target_vol: float) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, cov)
    cons = [
        cp.sum(w) == 1,
        w >= 0,
        risk <= target_vol**2
    ]
    prob = cp.Problem(cp.Maximize(ret), cons)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)
    if w.value is None:
        prob.solve(solver=cp.OSQP, verbose=False, max_iter=20000)
    if w.value is None:
        prob.solve(solver=cp.CVXOPT, verbose=False)
    if w.value is None:
        prob.solve(verbose=False)
    if w.value is None:
        raise SystemExit("[ERROR] Optimization failed to converge.")
    sol = np.maximum(w.value, 0.0)
    s = sol.sum()
    return sol / s if s > 0 else sol

def max_sharpe(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov)
    eps = 1e-8
    obj = (mu @ w) - eps * cp.sum_squares(w)
    cons = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)
    if w.value is None:
        prob.solve(solver=cp.OSQP, verbose=False, max_iter=20000)
    if w.value is None:
        prob.solve(verbose=False)
    if w.value is None:
        raise SystemExit("[ERROR] Sharpe optimization failed.")
    sol = np.maximum(w.value, 0.0)
    s = sol.sum()
    return sol / s if s > 0 else sol

def main():
    args = parse_args()

    df = read_holdings(args.holdings)
    df = infer_sleeves(df)

    r = load_returns(args.returns_file)

    w_sleeve = weights_by_sleeve(df)

    common, mu, cov = normalize_inputs(w_sleeve, r)
    if common is None:
        raise SystemExit(1)

    if args.target_vol is not None:
        w = max_return_at_vol(mu, cov, float(args.target_vol))
    else:
        w = max_sharpe(mu, cov)

    out = pd.Series(w, index=common, name="Weight")
    out = (out / out.sum()).sort_values(ascending=False)

    port_mu = float(np.dot(mu, w))
    port_vol = float(np.sqrt(w @ cov @ w))
    sharpe = port_mu / port_vol if port_vol > 0 else np.nan

    print("Weights (%):")
    print((out * 100).round(2).to_frame().T if len(out) < 20 else (out * 100).round(2))
    print("")
    print(f"Expected Return %: {(port_mu*100):.2f}")
    print(f"Volatility %: {(port_vol*100):.2f}")
    print(f"Sharpe: {sharpe:.2f}")

if __name__ == "__main__":
    main()