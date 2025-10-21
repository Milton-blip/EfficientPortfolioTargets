#!/usr/bin/env python3
import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Utilities ----------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_csv_required(path: Path, desc: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] Could not find {desc}. Expected at: {path}")
        sys.exit(1)
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] Failed reading {desc} at {path}: {e}")
        sys.exit(1)


# ---------- Sleeve map & assignment ----------

def load_sleeve_map(path: Path) -> pd.DataFrame | None:
    """
    Load portfolio_data/sleeve_map.csv if present.
    Expected columns: Symbol, Sleeve
    """
    if not path.exists():
        print(f"[WARN] Sleeve map not found at {path}. Will rely on existing 'Sleeve' values in holdings.")
        return None
    df = pd.read_csv(path)
    needed = {"Symbol", "Sleeve"}
    missing = needed.difference(df.columns)
    if missing:
        print(f"[ERROR] sleeve_map.csv is missing columns: {sorted(missing)}")
        sys.exit(1)
    df = df.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df["Sleeve"] = df["Sleeve"].astype(str).str.strip()
    return df


def assign_sleeves(holdings: pd.DataFrame, sleeve_map: pd.DataFrame | None) -> pd.DataFrame:
    """
    If holdings['Sleeve'] is null/empty, fill from sleeve_map (by Symbol).
    """
    df = holdings.copy()

    # Normalize symbol and sleeve fields
    if "Symbol" not in df.columns:
        print("[ERROR] Holdings missing required column: 'Symbol'")
        sys.exit(1)

    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()

    if "Sleeve" not in df.columns:
        df["Sleeve"] = ""

    # Treat empty strings as NaN for fill logic, then restore strings
    df["Sleeve"] = df["Sleeve"].astype(str)
    df["Sleeve"] = df["Sleeve"].replace({"": np.nan})

    if sleeve_map is not None:
        sym_to_sleeve = (
            sleeve_map
            .dropna(subset=["Symbol", "Sleeve"])
            .drop_duplicates(subset=["Symbol"])
            .set_index("Symbol")["Sleeve"]
            .to_dict()
        )
        need_fill = df["Sleeve"].isna()
        if need_fill.any():
            df.loc[need_fill, "Sleeve"] = df.loc[need_fill, "Symbol"].map(sym_to_sleeve)

    # Any still missing → set to NaN (exclude later if no returns exist)
    df["Sleeve"] = df["Sleeve"].where(~df["Sleeve"].isna(), np.nan)

    return df


# ---------- Data normalization ----------

def compute_holdings_weights_by_sleeve(holdings: pd.DataFrame) -> pd.Series:
    """
    Aggregate holdings by Sleeve (using MarketValue if present, else Quantity*PricePerShare),
    then convert to weight per sleeve (sum=1).
    """
    df = holdings.copy()

    for col in ["Quantity", "PricePerShare", "MarketValue"]:
        if col not in df.columns:
            df[col] = np.nan

    # compute MarketValue if missing or invalid
    mv = df["MarketValue"].astype(float)
    if mv.isna().all() or (mv <= 0).all():
        # try compute from quantity*price
        q = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0.0)
        p = pd.to_numeric(df["PricePerShare"], errors="coerce").fillna(0.0)
        df["MarketValue"] = q * p
    else:
        df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce").fillna(0.0)

    # require Sleeve for grouping
    if "Sleeve" not in df.columns:
        print("[ERROR] Holdings missing 'Sleeve' column after assignment.")
        sys.exit(1)

    # Exclude rows with no sleeve or zero/negative MV
    df = df.dropna(subset=["Sleeve"])
    df = df[df["MarketValue"] > 0]

    if df.empty:
        print("[ERROR] No valid (Sleeve, MarketValue) rows in holdings after cleaning.")
        sys.exit(1)

    sleeve_mv = df.groupby("Sleeve")["MarketValue"].sum()
    total_mv = sleeve_mv.sum()
    if total_mv <= 0:
        print("[ERROR] Total MarketValue is zero or negative after cleaning.")
        sys.exit(1)

    return (sleeve_mv / total_mv).sort_index()


def load_returns_wide(path: Path) -> pd.DataFrame:
    """
    Returns CSV in wide format: Date, <Sleeve1>, <Sleeve2>, ...
    """
    r = load_csv_required(path, "returns file")
    if "Date" not in r.columns:
        print("[ERROR] Returns file must have a 'Date' column.")
        sys.exit(1)
    # ensure numeric for sleeves
    for c in r.columns:
        if c == "Date":
            continue
        r[c] = pd.to_numeric(r[c], errors="coerce")
    # drop rows that are all-NaN across sleeves
    value_cols = [c for c in r.columns if c != "Date"]
    r = r.dropna(subset=value_cols, how="all")
    if r.empty:
        print("[ERROR] Returns file has no usable rows after cleanup.")
        sys.exit(1)
    return r


def align_to_overlap(holdings_w: pd.Series, returns_wide: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Keep only sleeves that exist in both holdings weights and returns columns.
    Return (mu, holdings_w_aligned, cov) where:
      - mu: mean returns (vector) for overlapping sleeves (index match)
      - holdings_w_aligned: holdings weights restricted to overlapping sleeves and renormalized
      - cov: covariance matrix for overlapping sleeves (index/columns match)
    """
    sleeves_returns = [c for c in returns_wide.columns if c != "Date"]
    sleeves_holdings = list(holdings_w.index)

    overlap = sorted(set(sleeves_returns).intersection(sleeves_holdings))
    if not overlap:
        print("[WARN] After normalization, no sleeves overlap between holdings and returns.")
        debug = {
            "holdings_sleeves": sleeves_holdings,
            "returns_sleeves": sleeves_returns
        }
        Path("debug_no_overlap.json").write_text(json.dumps(debug, indent=2))
        sys.exit(1)

    R = returns_wide[overlap].dropna(how="all")
    # remove columns that are entirely NaN just in case
    all_nan_cols = [c for c in overlap if R[c].notna().sum() == 0]
    if all_nan_cols:
        print(f"[NOTE] Excluding sleeves with no return history from stats: {all_nan_cols}")
        keep = [c for c in overlap if c not in all_nan_cols]
        if not keep:
            print("[ERROR] No sleeves remain after excluding empty return columns.")
            sys.exit(1)
        overlap = keep
        R = returns_wide[overlap].dropna(how="all")

    mu = R.mean().astype(float)
    cov = R.cov().astype(float)

    # align holdings weights and renormalize on overlap
    w = holdings_w.reindex(overlap).fillna(0.0)
    if w.sum() <= 0:
        # if all were zero (e.g., holdings all in sleeves lacking returns), fall back to uniform
        w = pd.Series(1.0 / len(overlap), index=overlap)
    else:
        w = w / w.sum()

    return mu, w, cov


# ---------- Optimization ----------

def solve_max_return_at_vol(mu: pd.Series,
                            cov: pd.DataFrame,
                            target_vol: float,
                            current_w: pd.Series | None = None,
                            max_weight: float | None = None,
                            min_weight: float | None = None,
                            l2_to_current: float | None = None) -> pd.Series:
    """
    Maximize mu^T w subject to:
      1) sum(w) = 1, w >= 0
      2) w^T cov w <= (target_vol)^2
      3) optional bounds and L2 penalty to current weights (convex)
    """
    sleeves = list(mu.index)
    n = len(sleeves)
    if cov.shape != (n, n):
        cov = cov.reindex(index=sleeves, columns=sleeves).fillna(0.0)

    Q = cov.values
    m = mu.values
    w = cp.Variable(n, nonneg=True)

    obj = cp.Maximize(m @ w)

    cons = [cp.sum(w) == 1, cp.quad_form(w, Q) <= float(target_vol) ** 2]

    if max_weight is not None:
        cons.append(w <= float(max_weight))
    if min_weight is not None:
        cons.append(w >= float(min_weight))

    reg = 0
    if l2_to_current is not None and current_w is not None:
        cur = current_w.reindex(sleeves).fillna(0.0).values
        reg = float(l2_to_current) * cp.sum_squares(w - cur)

    prob = cp.Problem(cp.Maximize(m @ w - reg), cons)
    # Prefer ECOS or OSQP; fallback to SCS
    try:
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=20000)
    except Exception:
        try:
            prob.solve(solver=cp.OSQP, verbose=False, max_iter=40000)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=40000)

    if w.value is None:
        print("[ERROR] Optimization failed to produce a solution.")
        sys.exit(1)

    sol = pd.Series(np.maximum(w.value, 0.0), index=sleeves)
    if sol.sum() <= 0:
        print("[ERROR] Optimizer returned all-zero weights.")
        sys.exit(1)
    sol = sol / sol.sum()
    return sol


# ---------- Reporting & plots ----------

def print_report(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame):
    exp_ret = float(mu.reindex(weights.index).fillna(0.0).values @ weights.values)
    vol = float(np.sqrt(weights.values @ cov.reindex(index=weights.index, columns=weights.index).fillna(0.0).values @ weights.values))
    sharpe = exp_ret / vol if vol > 0 else np.nan

    dfw = (weights * 100).to_frame("Weight%").sort_values("Weight%", ascending=False)
    print("\nWeights (%):")
    print(dfw.to_string(float_format=lambda x: f"{x:0.2f}"))

    print(f"\nExpected Return %: {exp_ret*100:0.2f}")
    print(f"Volatility %: {vol*100:0.2f}")
    if np.isfinite(sharpe):
        print(f"Sharpe: {sharpe:0.2f}")
    else:
        print("Sharpe: n/a")

    # mean returns by sleeve (period), just to aid sanity-checks
    print("\nMean period returns by sleeve (%), descending:")
    print((mu.sort_values(ascending=False) * 100).round(3))


def save_outputs(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, outputs_dir: Path):
    ensure_dir(outputs_dir)

    # 1) weights CSV
    (weights.to_frame("weight")
     .to_csv(outputs_dir / "weights.csv", index=True))

    # 2) correlation heatmap
    corr = cov.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    d = np.sqrt(np.diag(cov.reindex(index=weights.index, columns=weights.index).fillna(0.0).values))
    d[d == 0] = 1.0
    Dinv = np.diag(1.0 / d)
    corr = pd.DataFrame(Dinv @ cov.reindex(index=weights.index, columns=weights.index).fillna(0.0).values @ Dinv,
                        index=weights.index, columns=weights.index)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Sleeve Correlation")
    plt.tight_layout()
    plt.savefig(outputs_dir / "correlation.png", dpi=160)
    plt.close()

    # 3) weights bar plot
    plt.figure(figsize=(8, 5))
    (weights.sort_values(ascending=False) * 100.0).plot(kind="bar")
    plt.ylabel("Weight (%)")
    plt.title("Optimized Sleeve Weights")
    plt.tight_layout()
    plt.savefig(outputs_dir / "weights.png", dpi=160)
    plt.close()


# ---------- CLI / Main ----------

def parse_args():
    p = argparse.ArgumentParser(description="EfficientPortfolioTargets — maximize expected return at a target volatility.")
    p.add_argument("--holdings", required=True, help="Path to holdings CSV.")
    p.add_argument("--target-vol", required=True, type=float, help="Target volatility (e.g., 0.08 for 8%%).")
    p.add_argument("--returns-file", default="returns/sleeve_returns.csv", help="Path to sleeve returns CSV.")
    p.add_argument("--max-weight", type=float, default=None, help="Optional per-sleeve maximum weight (e.g., 0.35).")
    p.add_argument("--min-weight", type=float, default=None, help="Optional per-sleeve minimum weight (e.g., 0.00).")
    p.add_argument("--l2-to-current", type=float, default=None, help="Optional L2 penalty strength to current weights (e.g., 0.25).")
    p.add_argument("--outputs-dir", default="outputs", help="Directory to write outputs into.")
    return p.parse_args()


def main():
    args = parse_args()

    outputs_dir = Path(args.outputs_dir)
    ensure_dir(outputs_dir)

    holdings_path = Path(args.holdings)
    returns_path = Path(args.returns_file)
    sleeve_map_path = Path("portfolio_data/sleeve_map.csv")

    # Load inputs
    holdings_raw = load_csv_required(holdings_path, "holdings file")
    sleeve_map = load_sleeve_map(sleeve_map_path)
    holdings = assign_sleeves(holdings_raw, sleeve_map)

    # Compute holdings weights by sleeve
    current_w = compute_holdings_weights_by_sleeve(holdings)

    # Load returns and align to overlap
    returns_wide = load_returns_wide(returns_path)
    mu, w_aligned, cov = align_to_overlap(current_w, returns_wide)

    # Solve optimization
    weights = solve_max_return_at_vol(
        mu=mu,
        cov=cov,
        target_vol=float(args.target_vol),
        current_w=w_aligned,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        l2_to_current=args.l2_to_current
    )

    # Report and outputs
    print_report(weights, mu, cov)
    save_outputs(weights, mu, cov, outputs_dir=outputs_dir)


if __name__ == "__main__":
    main()