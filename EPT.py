#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RETURNS = PROJECT_ROOT / "returns" / "sleeve_returns.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SLEEVE_MAP_CSV = PROJECT_ROOT / "portfolio_data" / "sleeve_map.csv"


def ensure_outputs_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def canon(x) -> str:
    return str(x).strip()


def load_sleeve_map(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(
            f"[ERROR] Could not find sleeve map: {csv_path}\n"
            f"Create it with headers: Symbol,Sleeve"
        )
    df = pd.read_csv(csv_path)
    need = {"Symbol", "Sleeve"}
    have = {c.strip() for c in df.columns}
    if not need.issubset(have):
        raise SystemExit(f"[ERROR] sleeve_map.csv missing columns: {sorted(need - have)}")

    df = df[[c for c in df.columns if c.strip() in need]].copy()
    df.columns = [c.strip() for c in df.columns]
    df["Symbol"] = df["Symbol"].map(canon)
    df["Sleeve"] = df["Sleeve"].map(canon)
    return df


def assign_sleeves(holdings: pd.DataFrame, sleeve_map: pd.DataFrame) -> pd.DataFrame:
    out = holdings.copy()
    for col in ["Symbol", "Sleeve"]:
        if col not in out.columns:
            out[col] = ""

    out["Symbol"] = out["Symbol"].map(canon)
    out["Sleeve"] = out["Sleeve"].astype(str).map(canon)

    # Fill missing sleeves from map
    sym2sleeve = dict(sleeve_map[["Symbol", "Sleeve"]].values)
    mask = (out["Sleeve"].eq("")) | (out["Sleeve"].str.lower().isin({"nan", "none"}))
    out.loc[mask, "Sleeve"] = out.loc[mask, "Symbol"].map(sym2sleeve).fillna("")

    return out


def load_returns(returns_csv: Path) -> pd.DataFrame:
    if not returns_csv.exists():
        raise FileNotFoundError(str(returns_csv))
    rf = pd.read_csv(returns_csv)
    if "Date" not in rf.columns:
        raise SystemExit("[ERROR] returns file must have a 'Date' column")
    return rf


def compute_current_sleeve_weights(holdings: pd.DataFrame) -> pd.Series:
    # Accept either MarketValue or Quantity*PricePerShare
    if "MarketValue" in holdings.columns:
        mv = pd.to_numeric(holdings["MarketValue"], errors="coerce").fillna(0.0)
    else:
        q = pd.to_numeric(holdings.get("Quantity", 0), errors="coerce").fillna(0.0)
        p = pd.to_numeric(holdings.get("PricePerShare", 0), errors="coerce").fillna(0.0)
        mv = q * p

    sleeves = holdings["Sleeve"].replace({"": np.nan}).dropna()
    mv = mv.loc[sleeves.index]
    by_sleeve = mv.groupby(sleeves).sum()
    total = float(by_sleeve.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    w = by_sleeve / total
    return w.sort_index()


def align_and_build_stats(holdings_sleeves: pd.Series, returns_df: pd.DataFrame):
    """Returns: mu (Series), cov (DataFrame), common_sleeves (Index)"""
    sleeves_hold = set(holdings_sleeves.index)
    ret_cols = [c for c in returns_df.columns if c != "Date"]
    sleeves_ret = set(ret_cols)
    common = sorted(sleeves_hold & sleeves_ret)

    if not common:
        # Dump debug file so user can see why
        dbg = {
            "holdings_sleeves": sorted(sleeves_hold),
            "returns_sleeves": sorted(sleeves_ret),
        }
        (OUTPUT_DIR / "debug_no_overlap.json").write_text(json.dumps(dbg, indent=2))
        raise SystemExit(
            "[ERROR] After normalization, no sleeves overlap between holdings and returns.\n"
            "       Wrote debugging info to outputs/debug_no_overlap.json"
        )

    R = returns_df[common].dropna().astype(float)
    mu = R.mean() * 100.0  # mean period returns in %
    cov = np.cov(R.values, rowvar=False)
    cov = pd.DataFrame(cov, index=common, columns=common)
    return mu, cov, pd.Index(common)


def solve_max_return_at_vol(mu: pd.Series, cov: pd.DataFrame, target_vol: float,
                            max_weight: float | None = None,
                            min_weight: float | None = None,
                            l2_to_current: float | None = None,
                            current_w: pd.Series | None = None) -> pd.Series:
    sleeves = mu.index.tolist()
    n = len(sleeves)
    if n == 0:
        raise SystemExit("[ERROR] No sleeves to optimize after alignment.")

    w = cp.Variable(n)
    mu_vec = mu.values / 100.0  # back to decimals
    cov_mat = cov.values

    expected_ret = mu_vec @ w
    variance = cp.quad_form(w, cov_mat)

    cons = [cp.sum(w) == 1, w >= 0]
    if max_weight is not None:
        cons.append(w <= max_weight)
    if min_weight is not None:
        cons.append(w >= min_weight)
    if target_vol is not None:
        cons.append(variance <= (target_vol ** 2))

    obj = cp.Maximize(expected_ret)

    if l2_to_current is not None and current_w is not None and len(current_w) == n:
        obj = cp.Maximize(expected_ret - l2_to_current * cp.sum_squares(w - current_w.values))

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)
    if w.value is None:
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=20000)
    if w.value is None:
        raise SystemExit("[ERROR] Optimization failed to converge with available solvers.")

    sol = pd.Series(np.clip(w.value, 0, None), index=sleeves)
    sol = sol / sol.sum()
    return sol


def print_results(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, target_vol: float | None):
    df = pd.DataFrame({"Weight%": (weights * 100.0).round(2)}).sort_values("Weight%", ascending=False)
    print("\nWeights (%):")
    print(df.to_string())

    # realized stats
    w = weights.values
    mu_vec = mu.reindex(weights.index).values / 100.0
    cov_mat = cov.reindex(index=weights.index, columns=weights.index).values

    exp_ret = float(mu_vec @ w) * 100.0
    vol = float(np.sqrt(w @ cov_mat @ w)) * 100.0
    sharpe = exp_ret / vol if vol > 0 else np.nan

    print(f"\nExpected Return %: {exp_ret:.2f}")
    print(f"Volatility %: {vol:.2f}")
    print(f"Sharpe: {sharpe:.2f}")

    # mean period returns printout
    print("\nMean period returns by sleeve (%), descending:")
    print(mu.sort_values(ascending=False).round(3))

    # Plot efficient frontier point (single point) for context
    ensure_outputs_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(vol, exp_ret, c="blue", label="Solution")
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Expected Return (%)")
    ax.set_title("Efficient Frontier – Solution Point")
    ax.legend()
    out_png = OUTPUT_DIR / "efficient_point.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"\nSaved figure: {out_png}")


def auto_generate_returns_if_needed(args, sleeves_in_holdings: list[str]) -> None:
    """Generate returns if missing or if requested due to concentration."""
    ret_csv = Path(args.returns_file) if args.returns_file else DEFAULT_RETURNS
    need_gen = False
    reason = None

    if not ret_csv.exists():
        need_gen = True
        reason = "missing"

    if need_gen:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "write_sample_returns.py"),
            "--sleeve-map", str(SLEEVE_MAP_CSV),
            "--months", "180",
            "--out", str(ret_csv),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[INFO] Auto-generated returns ({reason}): {ret_csv}")
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode(errors="ignore"))
            print(e.stderr.decode(errors="ignore"))
            raise SystemExit("[ERROR] Failed to auto-generate returns.")


# ----------------------------
# Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EfficientPortfolioTargets optimizer")
    p.add_argument("--holdings", required=True, help="Path to holdings CSV.")
    p.add_argument("--target-vol", type=float, required=True, help="Target volatility (e.g., 0.08 for 8%).")
    p.add_argument("--returns-file", default=str(DEFAULT_RETURNS), help="Path to returns CSV.")
    p.add_argument("--max-weight", type=float, default=None, help="Optional per-sleeve max weight (0–1).")
    p.add_argument("--min-weight", type=float, default=None, help="Optional per-sleeve min weight (0–1).")
    p.add_argument("--l2-to-current", type=float, default=None, help="Optional L2 penalty to current weights.")
    p.add_argument("--auto-regen", dest="auto_regen", action="store_true",
                   help="If set, auto-generate returns when missing or when solution is overly concentrated.")
    return p.parse_args()


def main():
    ensure_outputs_dir()
    args = parse_args()

    # Load holdings and assign sleeves dynamically from sleeve_map.csv
    if not Path(args.holdings).exists():
        raise SystemExit(f"[ERROR] Holdings file not found: {args.holdings}")

    holdings_raw = pd.read_csv(args.holdings)
    sleeve_map = load_sleeve_map(SLEEVE_MAP_CSV)
    holdings = assign_sleeves(holdings_raw, sleeve_map)

    # Compute current sleeve weights (for optional L2 anchoring)
    current_w = compute_current_sleeve_weights(holdings)
    sleeves_in_holdings = current_w.index.tolist()

    # Auto-generate returns if missing
    auto_generate_returns_if_needed(args, sleeves_in_holdings)

    # Load returns and align
    try:
        returns_df = load_returns(Path(args.returns_file))
    except FileNotFoundError:
        if args.auto_regen:
            auto_generate_returns_if_needed(args, sleeves_in_holdings)
            returns_df = load_returns(Path(args.returns_file))
        else:
            raise SystemExit(f"[ERROR] Could not find returns file. Expected at: {args.returns_file}")

    mu, cov, common = align_and_build_stats(current_w, returns_df)

    # Reindex current weights to common sleeves (fill missing with 0)
    current_w = current_w.reindex(common).fillna(0.0)

    # Solve optimization
    w = solve_max_return_at_vol(
        mu=mu.reindex(common),
        cov=cov.reindex(index=common, columns=common),
        target_vol=float(args.target_vol),
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        l2_to_current=args.l2_to_current,
        current_w=current_w if args.l2_to_current is not None else None,
    )

    # If overly concentrated and --auto-regen, regenerate and re-run once
    if args.auto_regen and float(w.max()) >= 0.98:
        print("[INFO] Solution overly concentrated; regenerating returns for better diversification...")
        auto_generate_returns_if_needed(args, sleeves_in_holdings)
        returns_df = load_returns(Path(args.returns_file))
        mu, cov, common = align_and_build_stats(current_w, returns_df)
        current_w = current_w.reindex(common).fillna(0.0)
        w = solve_max_return_at_vol(
            mu=mu.reindex(common),
            cov=cov.reindex(index=common, columns=common),
            target_vol=float(args.target_vol),
            max_weight=args.max_weight,
            min_weight=args.min_weight,
            l2_to_current=args.l2_to_current,
            current_w=current_w if args.l2_to_current is not None else None,
        )

    print_results(w, mu.reindex(w.index), cov.reindex(index=w.index, columns=w.index), args.target_vol)

    # Save a JSON snapshot
    snapshot = {
        "weights": {k: float(v) for k, v in w.items()},
        "target_vol": float(args.target_vol),
        "mu_percent": {k: float(v) for k, v in mu.reindex(w.index).items()},
    }
    (OUTPUT_DIR / "ept_last_run.json").write_text(json.dumps(snapshot, indent=2))
    print(f"Saved run snapshot: {OUTPUT_DIR / 'ept_last_run.json'}")


if __name__ == "__main__":
    main()