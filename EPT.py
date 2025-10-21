#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt


# ---------------------------
# Paths and small utilities
# ---------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RETURNS = PROJECT_ROOT / "returns" / "sleeve_returns.csv"
DEFAULT_SLEEVE_MAP = PROJECT_ROOT / "portfolio_data" / "sleeve_map.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TOOLS_DIR = PROJECT_ROOT / "tools"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def canon(x: str) -> str:
    return str(x).strip()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------
# Sleeve mapping & holdings
# ---------------------------

def load_sleeve_map(sleeve_map_path: Path) -> Dict[str, str]:
    if not sleeve_map_path.exists():
        raise SystemExit(f"[ERROR] Missing sleeve map: {sleeve_map_path}")
    sm = load_csv(sleeve_map_path).copy()
    need = {"Symbol", "Sleeve"}
    missing = need - set(sm.columns)
    if missing:
        raise SystemExit(f"[ERROR] sleeve_map.csv missing columns: {sorted(missing)}")
    sm["Symbol"] = sm["Symbol"].map(canon)
    sm["Sleeve"] = sm["Sleeve"].map(canon)
    return dict(sm[["Symbol", "Sleeve"]].values)


def assign_sleeves(holdings: pd.DataFrame, sleeve_map: Dict[str, str]) -> pd.DataFrame:
    df = holdings.copy()
    for col in ["Symbol", "Quantity", "PricePerShare", "MarketValue", "Sleeve"]:
        if col not in df.columns:
            raise SystemExit(f"[ERROR] Holdings missing required column: {col}")

    # Normalize types
    df["Symbol"] = df["Symbol"].map(canon)
    df["Sleeve"] = df["Sleeve"].astype(str).map(canon)

    # Backfill MarketValue if needed
    mv_null = df["MarketValue"].isna() | (df["MarketValue"] == 0)
    if mv_null.any():
        if {"Quantity", "PricePerShare"} <= set(df.columns):
            df.loc[mv_null, "MarketValue"] = (
                pd.to_numeric(df.loc[mv_null, "Quantity"], errors="coerce").fillna(0.0)
                * pd.to_numeric(df.loc[mv_null, "PricePerShare"], errors="coerce").fillna(0.0)
            )

    # Fill missing sleeves from map
    blank = (df["Sleeve"] == "") | df["Sleeve"].str.lower().isin({"nan", "none"})
    df.loc[blank, "Sleeve"] = df.loc[blank, "Symbol"].map(sleeve_map).fillna("")

    # Drop rows without sleeve or with zero MarketValue
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce").fillna(0.0)
    df = df[df["MarketValue"].abs() > 0.0].copy()
    df = df[df["Sleeve"].astype(str).str.len() > 0].copy()

    if df.empty:
        raise SystemExit("[ERROR] After inferring sleeves and filtering, no holdings remain.")

    return df


def sleeve_weights_from_holdings(df: pd.DataFrame) -> pd.Series:
    by_sleeve = df.groupby("Sleeve", as_index=True)["MarketValue"].sum().sort_values(ascending=False)
    total = by_sleeve.sum()
    if total <= 0:
        raise SystemExit("[ERROR] Total MarketValue is zero after filtering.")
    return by_sleeve / total


# ---------------------------
# Returns management
# ---------------------------

def call_write_sample_returns(sleeve_map_path: Path, out_csv: Path, months: int = 240, seed: int = 123) -> None:
    py = sys.executable
    script = TOOLS_DIR / "write_sample_returns.py"
    if not script.exists():
        raise SystemExit(f"[ERROR] Cannot generate returns; missing: {script}")
    ensure_dir(out_csv.parent)
    cmd = [
        py, str(script),
        "--sleeve-map", str(sleeve_map_path),
        "--months", str(months),
        "--seed", str(seed),
        "--out", str(out_csv),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit("[ERROR] write_sample_returns.py failed")


def load_or_generate_returns(returns_path: Path,
                             sleeve_map_path: Path,
                             needed_sleeves: List[str],
                             allow_generate: bool = True) -> pd.DataFrame:
    if not returns_path.exists():
        if allow_generate:
            call_write_sample_returns(sleeve_map_path, returns_path)
        else:
            raise SystemExit(f"[ERROR] Could not find returns file. Expected at: {returns_path}")

    rf = load_csv(returns_path).copy()
    if "Date" not in rf.columns:
        raise SystemExit("[ERROR] Returns file must include a 'Date' column.")
    # Keep only sleeves used by holdings
    keep_cols = ["Date"] + [c for c in rf.columns if c != "Date" and canon(c) in {canon(s) for s in needed_sleeves}]
    rf = rf.loc[:, keep_cols].copy()

    # Drop columns that are entirely NaN
    numeric_cols = [c for c in rf.columns if c != "Date"]
    for c in numeric_cols:
        rf[c] = pd.to_numeric(rf[c], errors="coerce")
    all_nan = [c for c in numeric_cols if rf[c].isna().all()]
    if all_nan:
        rf = rf.drop(columns=all_nan)

    # Require at least 2 sleeves
    numeric_cols = [c for c in rf.columns if c != "Date"]
    if len(numeric_cols) < 2:
        if allow_generate:
            # regenerate broader returns with more sleeves
            call_write_sample_returns(sleeve_map_path, returns_path, months=360, seed=777)
            rf = load_csv(returns_path).copy()
            keep_cols = ["Date"] + [c for c in rf.columns if c != "Date" and canon(c) in {canon(s) for s in needed_sleeves}]
            rf = rf.loc[:, keep_cols].copy()
        else:
            raise SystemExit("[ERROR] Returns insufficient: fewer than 2 sleeves overlap with holdings.")

    # Drop rows with all-NaN across sleeves
    numeric_cols = [c for c in rf.columns if c != "Date"]
    rf = rf.dropna(axis=0, how="all", subset=numeric_cols)
    if rf.empty:
        raise SystemExit("[ERROR] Returns file has no usable rows after cleaning.")
    return rf


def mean_and_cov_from_returns(rf: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    sleeves = [c for c in rf.columns if c != "Date"]
    R = rf[sleeves].astype(float)
    mu = R.mean(axis=0)          # average period return per sleeve
    cov = R.cov()                # covariance matrix
    return mu, cov


# ---------------------------
# Optimizer
# ---------------------------

def solve_max_return_at_vol(mu: pd.Series,
                            cov: pd.DataFrame,
                            target_vol: float,
                            min_w: float = 0.0,
                            max_w: float = 1.0,
                            l2_to_current: float = 0.0,
                            current_w: pd.Series = None) -> pd.Series:
    sleeves = list(mu.index)
    n = len(sleeves)
    if n < 2:
        raise SystemExit("[ERROR] Need at least 2 sleeves to optimize.")

    # Align shapes
    cov = cov.loc[sleeves, sleeves].values
    mu_vec = mu.values.reshape(-1)

    w = cp.Variable(n, nonneg=True)
    cons = [cp.sum(w) == 1.0]
    if min_w > 0:
        cons.append(w >= min_w)
    if max_w < 1.0:
        cons.append(w <= max_w)

    risk = cp.quad_form(w, cov)               # variance
    cons.append(risk <= float(target_vol) ** 2)

    objective = mu_vec @ w
    if l2_to_current and current_w is not None:
        curr = current_w.reindex(sleeves).fillna(0.0).values.reshape(-1)
        objective = objective - float(l2_to_current) * cp.sum_squares(w - curr)

    prob = cp.Problem(cp.Maximize(objective), cons)

    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=20000)

    if w.value is None:
        raise SystemExit("[ERROR] Optimization failed to find a solution.")

    out = pd.Series(w.value, index=sleeves)
    out[out < 0] = 0
    s = out.sum()
    if s <= 0:
        raise SystemExit("[ERROR] Optimizer returned zero weights.")
    out /= s
    return out


# ---------------------------
# Diagnostics and plotting
# ---------------------------

def summarize_and_save(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, tag: str) -> None:
    ensure_dir(OUTPUT_DIR)

    sleeves = list(weights.index)
    w = weights.reindex(sleeves).fillna(0.0).values.reshape(-1)
    mu_vec = mu.reindex(sleeves).fillna(0.0).values.reshape(-1)
    cov_m = cov.loc[sleeves, sleeves].values

    exp_ret = float(mu_vec @ w) * 100.0
    vol = float(np.sqrt(w @ cov_m @ w)) * 100.0
    sharpe = exp_ret / vol if vol > 0 else 0.0

    print("\nWeights (%):")
    dfw = (weights * 100.0).sort_values(ascending=False).to_frame("Weight%")
    print(dfw.round(2).to_string())

    print(f"\nExpected Return %: {exp_ret:.2f}")
    print(f"Volatility %: {vol:.2f}")
    print(f"Sharpe: {sharpe:.2f}")

    diag = {
        "weights": {k: float(v) for k, v in weights.items()},
        "expected_return_pct": exp_ret,
        "volatility_pct": vol,
        "sharpe": sharpe,
    }
    (OUTPUT_DIR / f"weights_{tag}.json").write_text(json.dumps(diag, indent=2))

    # Allocation bar chart
    plt.figure(figsize=(8, 4))
    (weights.sort_values(ascending=False) * 100.0).plot(kind="bar", color="#3b82f6")
    plt.ylabel("Weight (%)")
    plt.title(f"Optimized Allocation — {tag}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"allocation_{tag}.png", dpi=150)
    plt.close()


def quick_frontier(mu: pd.Series, cov: pd.DataFrame,
                   min_vol: float, max_vol: float,
                   steps: int = 9) -> pd.DataFrame:
    vols = np.linspace(min_vol, max_vol, steps)
    rows = []
    for v in vols:
        try:
            w = solve_max_return_at_vol(mu, cov, target_vol=float(v))
            sleeves = list(w.index)
            wv = w.values.reshape(-1)
            mu_vec = mu.reindex(sleeves).values.reshape(-1)
            cov_m = cov.loc[sleeves, sleeves].values
            rows.append({
                "target_vol": float(v),
                "achieved_vol": float(np.sqrt(wv @ cov_m @ wv)),
                "exp_ret": float(mu_vec @ wv),
            })
        except Exception:
            rows.append({"target_vol": float(v), "achieved_vol": np.nan, "exp_ret": np.nan})
    return pd.DataFrame(rows)


def plot_frontier(df: pd.DataFrame, tag: str) -> None:
    ensure_dir(OUTPUT_DIR)
    ok = df.dropna()
    if ok.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(ok["achieved_vol"] * 100.0, ok["exp_ret"] * 100.0, "-o", color="#10b981")
    plt.xlabel("Volatility (%)")
    plt.ylabel("Expected Return (%)")
    plt.title(f"Efficient Frontier — {tag}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"frontier_{tag}.png", dpi=150)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--holdings", required=True, help="Path to holdings CSV.")
    p.add_argument("--target-vol", required=True, type=float, help="Target volatility (e.g. 0.08 for 8%%).")
    p.add_argument("--returns-file", default=str(DEFAULT_RETURNS), help="Path to returns CSV.")
    p.add_argument("--sleeve-map", default=str(DEFAULT_SLEEVE_MAP), help="Path to sleeve_map.csv.")
    p.add_argument("--max-weight", type=float, default=1.0, help="Per-sleeve max weight (default 1.0).")
    p.add_argument("--min-weight", type=float, default=0.0, help="Per-sleeve min weight (default 0.0).")
    p.add_argument("--l2-to-current", type=float, default=0.0, help="Penalty to stay near current weights.")
    p.add_argument("--auto-regen", action="store_true", help="Regenerate returns if solution concentrates.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)

    holdings_path = Path(args.holdings)
    returns_path = Path(args.returns_file)
    sleeve_map_path = Path(args.sleeve_map)

    sleeve_map = load_sleeve_map(sleeve_map_path)
    holdings = load_csv(holdings_path)
    holdings = assign_sleeves(holdings, sleeve_map)
    cur_w = sleeve_weights_from_holdings(holdings)

    sleeves_needed = list(cur_w.index)

    rf = load_or_generate_returns(returns_path, sleeve_map_path, sleeves_needed, allow_generate=True)
    mu, cov = mean_and_cov_from_returns(rf)

    # Align mu/cov to holdings sleeves (intersection only)
    sleeves_overlap = [s for s in sleeves_needed if s in mu.index]
    if len(sleeves_overlap) < 2:
        # Regenerate broader returns and retry once
        call_write_sample_returns(sleeve_map_path, returns_path, months=360, seed=999)
        rf = load_csv(returns_path)
        mu, cov = mean_and_cov_from_returns(rf)
        sleeves_overlap = [s for s in sleeves_needed if s in mu.index]
        if len(sleeves_overlap) < 2:
            raise SystemExit("[ERROR] After regeneration, still fewer than 2 sleeves overlap.")

    mu = mu.reindex(sleeves_overlap)
    cov = cov.loc[sleeves_overlap, sleeves_overlap]
    cur_w = cur_w.reindex(sleeves_overlap).fillna(0.0)

    w = solve_max_return_at_vol(
        mu, cov,
        target_vol=float(args.target_vol),
        min_w=float(args.min_weight),
        max_w=float(args.max_weight),
        l2_to_current=float(args.l2_to_current),
        current_w=cur_w
    )

    # If overly concentrated and auto-regen requested, regenerate once and re-optimize
    if args.auto-regen and w.max() >= 0.98:
        call_write_sample_returns(sleeve_map_path, returns_path, months=480, seed=2025)
        rf = load_csv(returns_path)
        mu, cov = mean_and_cov_from_returns(rf)
        # keep same sleeves_overlap if present, otherwise reselect
        common = [s for s in sleeves_needed if s in mu.index]
        if len(common) >= 2:
            mu = mu.reindex(common)
            cov = cov.loc[common, common]
            cur_w = cur_w.reindex(common).fillna(0.0)
        w = solve_max_return_at_vol(
            mu, cov,
            target_vol=float(args.target_vol),
            min_w=float(args.min_weight),
            max_w=float(args.max_weight),
            l2_to_current=float(args.l2_to_current),
            current_w=cur_w
        )

    tag = f"vol_{int(round(float(args.target_vol)*100))}"
    summarize_and_save(w, mu, cov, tag=tag)

    # Frontier sweep for PNG
    fr = quick_frontier(mu, cov, min_vol=max(0.01, float(args.target_vol)*0.5), max_vol=float(args.target_vol)*1.5, steps=9)
    plot_frontier(fr, tag=tag)

    # Also dump correlation matrix to outputs for inspection
    corr = cov.copy()
    d = np.sqrt(np.clip(np.diag(cov.values), 1e-12, None))
    Dinv = np.diag(1.0 / d)
    corr.values[:] = Dinv @ cov.values @ Dinv
    corr.to_csv(OUTPUT_DIR / f"corr_{tag}.csv")


if __name__ == "__main__":
    main()