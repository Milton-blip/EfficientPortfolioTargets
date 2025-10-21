#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

# -----------------------------
# Paths / Output utilities
# -----------------------------
ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs"                     # CHANGED: outputs/
RETURNS_DEFAULT = ROOT / "returns" / "sleeve_returns.csv"
OUTDIR.mkdir(parents=True, exist_ok=True)

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def write_csv(df: pd.DataFrame, name: str) -> Path:
    p = OUTDIR / f"{name}"
    df.to_csv(p, index=True)
    return p

def write_json(obj, name: str) -> Path:
    p = OUTDIR / f"{name}"
    with p.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, (np.floating,)) else x,
        )
    return p

# -----------------------------
# Sleeve mapping / inference
# -----------------------------
# Minimal pragmatic map. Add as you see fit.
SYMBOL_TO_SLEEVE: Dict[str, str] = {
    # Cash / MMF
    "SPAXX": "Cash", "VMFXX": "Cash", "BIL": "Cash", "SGOV": "Treasuries",
    # Treasuries
    "SHY": "Treasuries", "VGSH": "Treasuries", "IEF": "Treasuries", "TLT": "Treasuries", "SPTL": "Treasuries",
    # TIPS
    "TIP": "TIPS", "SCHP": "TIPS", "VTIP": "TIPS",
    # US Core
    "VTI": "US_Core", "ITOT": "US_Core", "SCHB": "US_Core", "VOO": "US_Core", "IVV": "US_Core",
    # US Value / Growth / Small Value
    "VTV": "US_Value", "IWD": "US_Value",
    "IVW": "US_Growth", "VUG": "US_Growth",
    "VBR": "US_SmallValue", "VIOV": "US_SmallValue", "AVUV": "US_SmallValue",
    # Intl Developed
    "VEA": "Intl_DM", "IEFA": "Intl_DM",
    # Emerging
    "VWO": "EM", "IEMG": "EM",
    # IG Core Bonds
    "AGG": "IG_Core", "BND": "IG_Core",
    # Energy (example)
    "XLE": "Energy",
    # Hedged intl IG (optional)
    "BNDX": "IG_Intl_Hedged",
}

NAME_CUES: List[Tuple[str, str]] = [
    ("MONEY MARKET", "Cash"),
    ("TREAS", "Treasuries"),
    ("TIPS", "TIPS"),
    ("VALUE", "US_Value"),
    ("GROWTH", "US_Growth"),
    ("SMALL", "US_SmallValue"),
    ("EMERGING", "EM"),
    ("INTL", "Intl_DM"),
    ("INTERNATIONAL", "Intl_DM"),
    ("AGG", "IG_Core"),
    ("CORE BOND", "IG_Core"),
    ("ENERGY", "Energy"),
    ("ILLQ", "Illiquid_Automattic"),
]

def infer_sleeve(symbol: str, name: str) -> str:
    s = str(symbol or "").upper().strip()
    n = str(name or "").upper().strip()
    if s in SYMBOL_TO_SLEEVE:
        return SYMBOL_TO_SLEEVE[s]
    for cue, sleeve in NAME_CUES:
        if cue in n:
            return sleeve
    # default equity core if unknown
    return "US_Core"

# -----------------------------
# Holdings & returns ingest
# -----------------------------
def load_holdings(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # New schema you provided:
    # Symbol,Name,Account,TaxStatus,Quantity,PricePerShare,MarketValue,
    # CostPerShare,TotalCost,Sleeve,Tradable,Notes
    req = ["Symbol", "Name", "Quantity", "PricePerShare", "MarketValue", "Sleeve"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] Holdings missing required columns: {missing}")

    # Ensure numeric
    for c in ["Quantity", "PricePerShare", "MarketValue"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Sleeve fill: use existing if present, otherwise infer
    sleeves = df["Sleeve"].astype(object)
    mask = sleeves.isna() | (sleeves.astype(str).str.strip() == "")
    if mask.any():
        inferred = [
            infer_sleeve(sym, nm) if m else sleeves.iloc[i]
            for i, (m, sym, nm) in enumerate(
                zip(mask.tolist(), df["Symbol"].tolist(), df["Name"].tolist())
            )
        ]
        sleeves.loc[mask] = inferred
    df["Sleeve"] = sleeves.astype(str)

    # Total sleeve dollars from actual market value
    sleeve_val = df.groupby("Sleeve")["MarketValue"].sum().replace({np.nan: 0.0})
    total = float(sleeve_val.sum())
    if total <= 0:
        raise SystemExit("[ERROR] Holdings have zero total MarketValue.")

    sleeve_weights = sleeve_val / total
    return df, sleeve_weights

def load_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            "[ERROR] Could not find returns file.\n"
            "Place a CSV like:\n"
            "  returns/sleeve_returns.csv\n"
            "with columns: Date,<sleeve1>,<sleeve2>,...\n"
            "Each column should be period returns (e.g., monthly decimal returns)."
        )
    r = pd.read_csv(path)

    # Expect wide format: Date, Sleeve1, Sleeve2, ...
    # Drop Date if present
    if "Date" in r.columns:
        r = r.drop(columns=["Date"])

    # Coerce to numeric
    r = r.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are entirely NaN
    r = r.dropna(axis=1, how="all")
    # Drop rows with all NaN
    r = r.dropna(axis=0, how="all")
    return r

def normalize_inputs(
    holdings_w: pd.Series,
    returns_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    # Overlap sleeves
    sleeves_h = set(holdings_w.index)
    sleeves_r = set(returns_df.columns)
    overlap = sorted(sleeves_h.intersection(sleeves_r))
    if not overlap:
        dbg = {
            "holdings_sleeves": sorted(list(sleeves_h)),
            "returns_sleeves": sorted(list(sleeves_r)),
        }
        write_json(dbg, "debug_no_overlap.json")
        raise SystemExit(
            "[WARN] After normalization, no sleeves overlap between holdings and returns.\n"
            "      Wrote debugging info to outputs/debug_no_overlap.json"
        )

    # Filter and renorm holdings to overlap
    h = holdings_w.reindex(overlap).fillna(0.0)
    total = float(h.sum())
    if total <= 0:
        # if all zero in overlap, put equal weights over overlap
        h = pd.Series(1.0 / len(overlap), index=overlap)

    # Returns subset
    R = returns_df[overlap].dropna(how="any")

    # Stats
    mu = R.mean()      # arithmetic mean per period
    cov = R.cov()
    return h, mu, cov

# -----------------------------
# Optimization (convex)
# Efficient-frontier bisection:
# minimize variance subject to return >= r,
# then bisection on r to hit target variance.
# -----------------------------
def min_variance_with_return(
    mu: pd.Series,
    cov: pd.DataFrame,
    r_target: float,
    wmin: float,
    wmax: float,
    w0: pd.Series | None = None
) -> Tuple[np.ndarray, float]:
    cols = mu.index.tolist()
    n = len(cols)
    Sigma = cov.values

    # PSD guard: small jitter (helps OSQP/SCS; keeps convexity)
    eps = 1e-10
    Sigma = Sigma + eps * np.eye(n)

    w = cp.Variable(n, nonneg=True)
    objective = cp.quad_form(w, Sigma)
    cons = [
        cp.sum(w) == 1.0,
        mu.values @ w >= r_target,
    ]
    if wmin is not None:
        cons.append(w >= float(wmin))
    if wmax is not None:
        cons.append(w <= float(wmax))

    prob = cp.Problem(cp.Minimize(objective), cons)

    solved = False
    for solver in (cp.OSQP, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, verbose=False, max_iters=20000)
            if w.value is not None:
                solved = True
                break
        except Exception:
            continue

    if not solved or w.value is None:
        raise RuntimeError("Min-variance QP failed to solve.")

    wv = np.clip(w.value, 0, None)
    s = float(wv.sum())
    if s <= 0:
        raise RuntimeError("Invalid weights solution (sum <= 0).")
    wv = wv / s

    var = float(wv.T @ Sigma @ wv)
    return wv, var

def frontier_at_vol(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_vol: float,
    wmin: float | None,
    wmax: float | None
) -> np.ndarray:
    """Find weights whose variance hits target_vol^2 (within tolerance) via bisection on expected return."""
    # Lower bound return: min-variance portfolio return
    w_minvar, var_min = min_variance_with_return(
        mu, cov, r_target=-1e9, wmin=wmin, wmax=wmax
    )
    r_low = float(mu.values @ w_minvar)

    # Upper bound return: maximum component return allowed by bounds
    r_high = float(mu.max())
    target_var = float(target_vol ** 2)

    # If min-var already above the risk budget, tighten with bounds or accept if cannot do lower
    if var_min > target_var:
        # Can’t reduce risk further under current bounds; return min-var
        return w_minvar

    best_w = w_minvar
    best_diff = abs(var_min - target_var)

    for _ in range(35):
        r_mid = 0.5 * (r_low + r_high)
        w_mid, var_mid = min_variance_with_return(
            mu, cov, r_target=r_mid, wmin=wmin, wmax=wmax
        )
        diff = var_mid - target_var
        if abs(diff) < best_diff:
            best_diff = abs(diff)
            best_w = w_mid
        # If we can afford more return (variance below target), move r_low up
        if var_mid <= target_var:
            r_low = r_mid
        else:
            r_high = r_mid

    return best_w

# -----------------------------
# Plotting: efficient frontier PNG
# -----------------------------
def render_frontier_png(
    mu: pd.Series,
    cov: pd.DataFrame,
    tag: str,
    targets: List[float] | None = None
) -> Path:
    """
    Sample a set of target vols, compute frontier points, and save a PNG under outputs/.
    """
    if targets is None:
        # a smooth set across 5%..20% vol
        targets = [x / 100.0 for x in range(5, 21, 1)]

    xs = []  # vol
    ys = []  # expected return
    for tv in targets:
        try:
            w = frontier_at_vol(mu, cov, tv, wmin=None, wmax=None)
            exp = float(mu.values @ w)
            vol = float(np.sqrt(w @ cov.values @ w))
            xs.append(vol)
            ys.append(exp)
        except Exception:
            # skip unsolved points
            continue

    if not xs:
        # nothing to plot
        return OUTDIR / f"frontier_{tag}_{_ts()}.png"  # still return a path

    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot([v * 100 for v in xs], [r * 100 for r in ys], marker="o", lw=1.5, ms=3)
    plt.xlabel("Volatility (%)")
    plt.ylabel("Expected Return (%)")
    plt.title(f"Efficient Frontier — {tag}")
    plt.grid(True, alpha=0.3)
    outpath = OUTDIR / f"frontier_{tag}_{_ts()}.png"
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return outpath

# -----------------------------
# Pretty printing
# -----------------------------
def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}"

def print_results(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, tag: str):
    w = weights.reindex(mu.index).fillna(0.0)
    exp = float(mu.values @ w.values)
    vol = float(np.sqrt(w.values @ cov.values @ w.values))
    sharpe = exp / vol if vol > 0 else np.nan

    print("\nWeights (%):")
    out = (w * 100).rename("Weight%").sort_values(ascending=False).to_frame()
    print(out.to_string())

    print(f"\nExpected Return %: {fmt_pct(exp)}")
    print(f"Volatility %: {fmt_pct(vol)}")
    print(f"Sharpe: {sharpe:.2f}")

    # Correlation
    std = np.sqrt(np.diag(cov.values))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov.values / np.outer(std, std)
    corr_df = pd.DataFrame(corr, index=cov.index, columns=cov.columns)
    print("\nCorrelation matrix:")
    print(corr_df.round(2).to_string())

    # Write artifacts
    ts = _ts()
    weights_path = write_csv(out, f"weights_{tag}_{ts}.csv")
    corr_path = write_csv(corr_df.round(6), f"corr_{tag}_{ts}.csv")
    meta = {
        "timestamp": ts,
        "tag": tag,
        "expected_return": exp,
        "volatility": vol,
        "sharpe": sharpe,
        "target_hit_note": "Bisection on return to achieve target variance boundary.",
        "weights_csv": str(weights_path),
        "correlation_csv": str(corr_path),
    }
    write_json(meta, f"run_{tag}_{ts}.json")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Efficient frontier optimizer on sleeves (outputs saved under outputs/).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--holdings", required=True, help="Path to holdings CSV.")
    p.add_argument("--returns-file", default=str(RETURNS_DEFAULT), help="Path to sleeve_returns.csv.")
    p.add_argument("--target-vol", type=float, required=True, help="Target volatility (e.g., 0.08 for 8%%).")
    p.add_argument("--max-weight", type=float, default=None, help="Per-sleeve max weight (e.g., 0.35).")
    p.add_argument("--min-weight", type=float, default=None, help="Per-sleeve min weight (e.g., 0.00).")
    p.add_argument("--l2-to-current", type=float, default=0.0, help="(Not used in frontier solve; reserved.)")
    return p.parse_args()

def main():
    args = parse_args()
    hold_path = Path(args.holdings).resolve()
    ret_path = Path(args.returns_file).resolve()

    # Load
    holdings_df, holdings_w = load_holdings(hold_path)
    returns_df = load_returns(ret_path)

    # Normalize by overlap
    current_w, mu, cov = normalize_inputs(holdings_w, returns_df)

    # Solve for weights that lie on the efficient frontier at target vol
    w_arr = frontier_at_vol(
        mu, cov,
        target_vol=float(args.target_vol),
        wmin=args.min_weight,
        wmax=args.max_weight
    )
    w_series = pd.Series(w_arr, index=mu.index)

    # Print & write
    tag = f"targetVol_{args.target_vol:.2f}"
    print_results(w_series, mu, cov, tag=tag)

    # PNG: efficient frontier curve
    png_path = render_frontier_png(mu, cov, tag=tag)
    print(f"\nFrontier PNG: {png_path}")

    # Validator tolerance check
    realized_vol = float(np.sqrt(w_series.values @ cov.values @ w_series.values))
    diff = abs(realized_vol - float(args.target_vol))
    if diff > 0.005:  # ±0.5% absolute vol tolerance
        print(
            f"\n[WARN] Achieved vol {fmt_pct(realized_vol)} differs from target {fmt_pct(args.target_vol)} "
            f"by {fmt_pct(diff)}. Tighten bounds/returns if you need closer matching."
        )

if __name__ == "__main__":
    main()