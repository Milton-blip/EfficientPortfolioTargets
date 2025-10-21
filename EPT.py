#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, List  # ← compat typing

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RETURNS = PROJECT_ROOT / "returns" / "sleeve_returns.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SLEEVE_MAP_CSV = PROJECT_ROOT / "portfolio_data" / "sleeve_map.csv"


def _infer_returns_label(returns_path: str) -> str:
    rp = str(returns_path).lower()
    if "real" in rp:
        return "Real"
    if "nominal" in rp:
        return "Nominal"
    return "Nominal"

def _resolve_return_label(return_type_opt: Optional[str], returns_path: str) -> str:
    """
    Prefer the --return-type flag; fall back to inferring from path.
    """
    if return_type_opt:
        return "Real" if return_type_opt.lower() == "real" else "Nominal"
    return _infer_returns_label(returns_path)

def to_real_returns(returns_df: pd.DataFrame, inflation_col: Optional[str] = None) -> pd.DataFrame:
    """
    Convert sleeve returns to REAL using (1+r)/(1+pi) - 1 per period.
    Expects returns_df columns: Date + sleeve columns, and an inflation column if available.
    If inflation_col is None, will try to find one named 'Inflation' or 'CPI' (case-insensitive).
    """
    df = returns_df.copy()
    cols = [c for c in df.columns if c.lower() not in {"date"}]

    # Find inflation column if not specified
    if inflation_col is None:
        for cand in df.columns:
            lc = cand.lower()
            if lc in {"inflation", "cpi"}:
                inflation_col = cand
                break

    if inflation_col is None or inflation_col not in df.columns:
        print("[WARN] No inflation column found; Real returns cannot be computed. Using nominal returns.")
        return df

    pi = pd.to_numeric(df[inflation_col], errors="coerce").fillna(0.0)
    sleeve_cols = [c for c in df.columns if c not in {"Date", inflation_col}]

    for c in sleeve_cols:
        r = pd.to_numeric(df[c], errors="coerce")
        df[c] = (1.0 + r) / (1.0 + pi) - 1.0

    return df

def plot_frontier_pretty(
    returns_path,
    frontier_vols,
    frontier_rets,
    mu_target=None,
    vol_target=None,
    mu_current=None,
    vol_current=None,
    mu_minvol=None,
    vol_minvol=None,
    mu_maxsharpe=None,
    vol_maxsharpe=None,
    scenario_name="Base",
    outputs_dir="outputs",
    label_override=None,  # <-- NEW
):
    """
    Produce a frontier chart. All inputs here are in DECIMALS (0.08 = 8%).
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    label_nv = label_override or _infer_returns_label(returns_path)

    plt.figure(figsize=(9.5, 5.6), dpi=140)

    # Frontier curve
    plt.plot(
        [v * 100 for v in frontier_vols],
        [r * 100 for r in frontier_rets],
        color="#0b245b",
        linewidth=3.0,
        label=f"Frontier ({label_nv})",
    )

    # Target vol point
    if mu_target is not None and vol_target is not None:
        plt.scatter(
            vol_target * 100,
            mu_target * 100,
            s=150,
            facecolors="gold",
            edgecolors="black",
            linewidths=1.2,
            zorder=4,
            label="Target 8% Vol",
        )

    # Max Sharpe
    if mu_maxsharpe is not None and vol_maxsharpe is not None:
        plt.scatter(
            vol_maxsharpe * 100,
            mu_maxsharpe * 100,
            s=110,
            facecolors="white",
            edgecolors="#0b8f2f",
            linewidths=2.0,
            zorder=4,
        )
        plt.scatter(
            vol_maxsharpe * 100,
            mu_maxsharpe * 100,
            s=110,
            facecolors="#0b8f2f",
            edgecolors="#0b8f2f",
            linewidths=1.2,
            zorder=4,
            label="MaxSharpe",
        )

    # Minimum volatility point
    if mu_minvol is not None and vol_minvol is not None:
        plt.scatter(
            vol_minvol * 100,
            mu_minvol * 100,
            s=110,
            facecolors="#1557ff",
            edgecolors="#0b245b",
            linewidths=1.2,
            zorder=4,
            label="MinVol",
        )

    # Current portfolio
    if mu_current is not None and vol_current is not None:
        plt.scatter(
            vol_current * 100,
            mu_current * 100,
            s=110,
            facecolors="#d32f2f",
            edgecolors="#8b1c1c",
            linewidths=1.2,
            zorder=4,
            label="Current",
        )

    plt.xlabel("Volatility (%)", fontsize=13)
    plt.ylabel("Expected Return (%)", fontsize=13)

    title = f"Efficient Frontier + 8% Target ({scenario_name}, {label_nv})"
    plt.title(title, fontsize=18, pad=12)

    lg = plt.legend(
        framealpha=0.9, facecolor="white", edgecolor="#cccccc", fontsize=12
    )
    # --- robust handle access across Matplotlib versions ---
    handles = getattr(lg, "legend_handles", None) or getattr(lg, "legendHandles", [])
    for lh in handles:
        if hasattr(lh, "set_linewidth"):
            lh.set_linewidth(2.0)

    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()

    safe_title = title.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    out_png = Path(outputs_dir) / f"{safe_title}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    return str(out_png)


def ensure_outputs_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def canon(x) -> str:
    return str(x).strip()


def load_sleeve_map(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(
            f"[ERROR] Could not find sleeve map: {csv_path}\nCreate it with headers: Symbol,Sleeve"
        )
    df = pd.read_csv(csv_path)
    need = {"Symbol", "Sleeve"}
    have = {c.strip() for c in df.columns}
    if not need.issubset(have):
        miss = sorted(need - have)
        raise SystemExit(f"[ERROR] sleeve_map.csv missing columns: {miss}")
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


def align_and_build_stats(holdings_w: pd.Series, returns_df: pd.DataFrame):
    sleeves_hold = set(holdings_w.index)
    ret_cols = [c for c in returns_df.columns if c != "Date"]
    sleeves_ret = set(ret_cols)
    common = sorted(sleeves_hold & sleeves_ret)
    if not common:
        ensure_outputs_dir()
        dbg = {
            "holdings_sleeves": sorted(sleeves_hold),
            "returns_sleeves": sorted(sleeves_ret),
        }
        (OUTPUT_DIR / "debug_no_overlap.json").write_text(json.dumps(dbg, indent=2))
        raise SystemExit(
            "[ERROR] After normalization, no sleeves overlap between holdings and returns.\n"
            "       Wrote debugging info to outputs/debug_no_overlap.json"
        )
    R = returns_df[common].dropna().astype(float)  # decimals
    mu = R.mean() * 100.0                           # percent
    cov = np.cov(R.values, rowvar=False)            # decimals^2
    cov = pd.DataFrame(cov, index=common, columns=common)
    return mu, cov, pd.Index(common)

def _solve_with_robust_settings(prob: cp.Problem) -> None:
    # Try ECOS (QP) with tighter tolerances, then OSQP, then SCS.
    try:
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=10000, abstol=1e-9, reltol=1e-9, feastol=1e-9)
    except Exception:
        pass
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or prob.value is None:
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=200000, polish=True, verbose=False)
        except Exception:
            pass
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or prob.value is None:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=200000, eps=1e-7)

def solve_max_return_at_vol(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_vol: float,
    max_weight: Optional[float] = None,
    min_weight: Optional[float] = None,
    l2_to_current: Optional[float] = None,
    current_w: Optional[pd.Series] = None,
) -> pd.Series:
    sleeves = mu.index.tolist()
    n = len(sleeves)
    if n == 0:
        raise SystemExit("[ERROR] No sleeves to optimize after alignment.")
    w = cp.Variable(n)
    mu_vec = mu.values / 100.0  # decimals
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
    _solve_with_robust_settings(prob)
    if w.value is None:
        raise SystemExit("[ERROR] Optimization failed to converge with available solvers.")
    sol = pd.Series(np.clip(w.value, 0, None), index=sleeves)
    return sol / sol.sum()


def solve_min_variance(
    cov: pd.DataFrame,
    max_weight: Optional[float] = None,
    min_weight: Optional[float] = None,
) -> pd.Series:
    n = cov.shape[0]
    w = cp.Variable(n)
    cov_mat = cov.values
    cons = [cp.sum(w) == 1, w >= 0]
    if max_weight is not None:
        cons.append(w <= max_weight)
    if min_weight is not None:
        cons.append(w >= min_weight)
    obj = cp.Minimize(cp.quad_form(w, cov_mat))
    prob = cp.Problem(obj, cons)
    _solve_with_robust_settings(prob)
    if w.value is None:
        raise SystemExit("[ERROR] Optimization failed to converge with available solvers.")
    s = pd.Series(np.clip(w.value, 0, None), index=cov.index)
    return s / s.sum()


def solve_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    max_weight: Optional[float] = None,
    min_weight: Optional[float] = None,
) -> pd.Series:
    # maximize mu@w  s.t. w'Σw <= 1, sum w = 1, w>=0
    n = len(mu)
    w = cp.Variable(n)
    mu_vec = mu.values / 100.0
    cov_mat = cov.values
    cons = [cp.sum(w) == 1, w >= 0, cp.quad_form(w, cov_mat) <= 1.0]
    if max_weight is not None:
        cons.append(w <= max_weight)
    if min_weight is not None:
        cons.append(w >= min_weight)
    obj = cp.Maximize(mu_vec @ w)
    prob = cp.Problem(obj, cons)
    _solve_with_robust_settings(prob)
    if w.value is None:
        raise SystemExit("[ERROR] Optimization failed to converge with available solvers.")
    s = pd.Series(np.clip(w.value, 0, None), index=mu.index)
    return s / s.sum()


def realized_stats(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame):
    w = weights.reindex(mu.index).values
    mu_vec = mu.values / 100.0
    cov_mat = cov.reindex(index=mu.index, columns=mu.index).values
    exp_ret = float(mu_vec @ w) * 100.0  # percent
    vol = float(np.sqrt(w @ cov_mat @ w)) * 100.0  # percent
    sharpe = exp_ret / vol if vol > 0 else np.nan
    return exp_ret, vol, sharpe

def frontier_by_gamma(
    mu: pd.Series,
    cov: pd.DataFrame,
    gamma_grid: np.ndarray,
    max_weight: Optional[float] = None,
    min_weight: Optional[float] = None,
) -> tuple[list, list]:
    """
    Trace the efficient frontier by sweeping risk aversion γ in:
        max_w  mu·w - γ * w'Σw  s.t. 1'w=1, w>=0, (bounds)
    Returns vols and rets in DECIMALS for plotting.
    """
    n = len(mu)
    w = cp.Variable(n)
    mu_vec = mu.values / 100.0
    cov_mat = cov.values
    cons = [cp.sum(w) == 1, w >= 0]
    if max_weight is not None:
        cons.append(w <= max_weight)
    if min_weight is not None:
        cons.append(w >= min_weight)

    vols, rets = [], []
    for g in gamma_grid:
        obj = cp.Maximize(mu_vec @ w - float(g) * cp.quad_form(w, cov_mat))
        prob = cp.Problem(obj, cons)
        _solve_with_robust_settings(prob)
        if w.value is None:
            continue
        ww = np.clip(w.value, 0, None)
        ww = ww / ww.sum()
        ws = pd.Series(ww, index=mu.index)
        r_pct, v_pct, _ = realized_stats(ws, mu, cov)
        rets.append(r_pct / 100.0)  # decimals
        vols.append(v_pct / 100.0)  # decimals

    if not vols:
        return [], []

    df = pd.DataFrame({"vol": vols, "ret": rets}).sort_values("vol").drop_duplicates("vol", keep="last")
    df["ret"] = df["ret"].cummax()   # upper envelope removes wiggles
    return df["vol"].tolist(), df["ret"].tolist()

def plot_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_w: pd.Series,
    current_w: Optional[pd.Series],
    ret_type_label: str,
    target_vol: float,
):
    ensure_outputs_dir()

    vols, rets = [], []
    grid = np.linspace(max(0.0025, 0.5 * target_vol), max(target_vol * 1.8, target_vol + 0.03), 40)
    for tv in grid:
        try:
            w_tv = solve_max_return_at_vol(mu, cov, tv)
            r, v, _ = realized_stats(w_tv, mu, cov)
            rets.append(r)
            vols.append(v)
        except Exception:
            pass

    minw = solve_min_variance(cov)
    min_r, min_v, _ = realized_stats(minw, mu, cov)

    msw = solve_max_sharpe(mu, cov)
    ms_r, ms_v, _ = realized_stats(msw, mu, cov)

    t_r, t_v, _ = realized_stats(target_w, mu, cov)

    cur_r = cur_v = None
    if current_w is not None and len(current_w) == len(mu.index):
        cr, cv, _ = realized_stats(current_w, mu, cov)
        cur_r, cur_v = cr, cv

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if vols and rets:
        ax.plot(vols, rets, color="navy", lw=2.0, label=f"Frontier ({ret_type_label})")

    ax.scatter([t_v], [t_r], s=80, edgecolors="black", facecolors="gold", label="Target 8 percent Vol")
    ax.scatter([ms_v], [ms_r], s=40, color="green", label="MaxSharpe")
    ax.scatter([min_v], [min_r], s=40, color="blue", label="MinVol")
    if cur_r is not None and cur_v is not None:
        ax.scatter([cur_v], [cur_r], s=40, color="red", label="Current")

    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Expected Return (%)")
    ax.set_title(f"Efficient Frontier + 8 percent Target (Base, {ret_type_label})")
    ax.legend(loc="best", frameon=True)
    ax.grid(False)
    fig.tight_layout()
    out_png = OUTPUT_DIR / f"efficient_frontier_target_{int(round(target_vol*100))}.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved frontier figure: {out_png}")


def print_results(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, target_vol: float, ret_label: str):
    df = pd.DataFrame({"Weight%": (weights * 100.0).round(2)}).sort_values("Weight%", ascending=False)
    print("\nWeights (%):")
    print(df.to_string())
    r, v, s = realized_stats(weights, mu, cov)
    print(f"\nExpected Return % ({ret_label}): {r:.2f}")
    print(f"Volatility %: {v:.2f}")
    print(f"Sharpe ({ret_label}): {s:.2f}")
    print("\nMean period returns by sleeve (% " + ret_label + "), descending:")
    print(mu.sort_values(ascending=False).round(3))


def auto_generate_returns_if_needed(args, sleeves_in_holdings: List[str]) -> None:
    ret_csv = Path(args.returns_file) if args.returns_file else DEFAULT_RETURNS
    need_gen = False
    if not ret_csv.exists():
        need_gen = True
    if need_gen:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "write_sample_returns.py"),
            "--sleeve-map",
            str(SLEEVE_MAP_CSV),
            "--months",
            "180",
            "--out",
            str(ret_csv),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"[INFO] Auto-generated returns: {ret_csv}")
        except subprocess.CalledProcessError as e:
            try:
                print(e.stdout.decode(errors="ignore"))
                print(e.stderr.decode(errors="ignore"))
            except Exception:
                pass
            raise SystemExit("[ERROR] Failed to auto-generate returns.")


def parse_args():
    p = argparse.ArgumentParser(description="EfficientPortfolioTargets optimizer")
    p.add_argument("--holdings", required=True, help="Path to holdings CSV.")
    p.add_argument(
        "--target-vol",
        type=float,
        required=True,
        help="Target volatility (example: 0.08 for 8 percent).",
    )
    p.add_argument("--returns-file", default=str(DEFAULT_RETURNS), help="Path to returns CSV.")
    p.add_argument("--max-weight", type=float, default=None, help="Optional per-sleeve max weight (0–1).")
    p.add_argument("--min-weight", type=float, default=None, help="Optional per-sleeve min weight (0–1).")
    p.add_argument("--l2-to-current", type=float, default=None, help="Optional L2 penalty to current weights.")
    p.add_argument(
        "--auto-regen",
        dest="auto_regen",
        action="store_true",
        help="Auto-generate returns if missing and retry if solution is overly concentrated.",
    )
    p.add_argument(
        "--return-type",
        choices=["nominal", "real"],
        default="nominal",
        help="Used only for labeling charts: nominal or real.",
    )
    return p.parse_args()


def main():
    ensure_outputs_dir()
    args = parse_args()

    ret_label = _resolve_return_label(getattr(args, "return_type", None), args.returns_file)

    if not Path(args.holdings).exists():
        raise SystemExit(f"[ERROR] Holdings file not found: {args.holdings}")

    holdings_raw = pd.read_csv(args.holdings)
    sleeve_map = load_sleeve_map(SLEEVE_MAP_CSV)
    holdings = assign_sleeves(holdings_raw, sleeve_map)
    current_w = compute_current_sleeve_weights(holdings)
    sleeves_in_holdings = current_w.index.tolist()

    auto_generate_returns_if_needed(args, sleeves_in_holdings)

    try:
        returns_df = load_returns(Path(args.returns_file))
        # Convert to real returns if requested
        ret_label = _resolve_return_label(getattr(args, "return_type", None), args.returns_file)
        if ret_label == "Real":
            returns_df = to_real_returns(returns_df, inflation_col=None)  # auto-detect 'Inflation'/'CPI'
    except FileNotFoundError:
        if args.auto_regen:
            auto_generate_returns_if_needed(args, sleeves_in_holdings)
            returns_df = load_returns(Path(args.returns_file))
        else:
            raise SystemExit(f"[ERROR] Could not find returns file. Expected at: {args.returns_file}")

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

    print_results(w, mu.reindex(w.index), cov.reindex(index=w.index, columns=w.index), args.target_vol, ret_label)

    # --- build frontier via gamma-sweep for a smooth/convex curve (inputs to pretty plot are DECIMALS) ---
    gamma_grid = np.logspace(-1, 3, 120)  # sweep risk aversion over several orders
    vols_on_grid, rets_on_grid = frontier_by_gamma(
        mu.reindex(common),
        cov.reindex(index=common, columns=common),
        gamma_grid=gamma_grid,
        max_weight=args.max_weight,
        min_weight=args.min_weight,
    )

    # Key points (percent from realized_stats → convert to decimals for pretty plot)
    w_min = solve_min_variance(cov)
    mu_minvol, vol_minvol, _ = realized_stats(w_min, mu, cov)

    w_ms = solve_max_sharpe(mu, cov)
    mu_maxsharpe, vol_maxsharpe, _ = realized_stats(w_ms, mu, cov)

    mu_at_target, vol_at_target, _ = realized_stats(w, mu, cov)

    mu_current = vol_current = None
    if current_w is not None and len(current_w) == len(mu.index):
        mu_current, vol_current, _ = realized_stats(current_w, mu, cov)

    png_path = plot_frontier_pretty(
        returns_path=args.returns_file or "returns/sleeve_returns.csv",
        frontier_vols=vols_on_grid,                 # decimals
        frontier_rets=rets_on_grid,                 # decimals
        mu_target=mu_at_target / 100.0,             # decimals
        vol_target=vol_at_target / 100.0,           # decimals
        mu_current=(mu_current / 100.0) if mu_current is not None else None,
        vol_current=(vol_current / 100.0) if vol_current is not None else None,
        mu_minvol=mu_minvol / 100.0,
        vol_minvol=vol_minvol / 100.0,
        mu_maxsharpe=mu_maxsharpe / 100.0,
        vol_maxsharpe=vol_maxsharpe / 100.0,
        scenario_name="Base",
        outputs_dir=str(OUTPUT_DIR),
        label_override=ret_label,  # <-- NEW
    )
    print(f"Saved efficient frontier chart: {png_path}")

    snapshot = {
        "weights": {k: float(v) for k, v in w.items()},
        "target_vol": float(args.target_vol),
        "mu_percent": {k: float(v) for k, v in mu.reindex(w.index).items()},
        "return_label": ret_label,
    }
    (OUTPUT_DIR / "ept_last_run.json").write_text(json.dumps(snapshot, indent=2))
    print(f"Saved run snapshot: {OUTPUT_DIR / 'ept_last_run.json'}")


if __name__ == "__main__":
    main()