#!/usr/bin/env python3
# PA2.py — Efficient portfolio targets with robust schema + name normalization
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import math
import re
import numpy as np
import pandas as pd

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False


# -------------------------------
# CLI
# -------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EfficientPortfolioTargets (schema-flexible + name normalization)")
    p.add_argument(
        "--holdings",
        default="portfolio_data/holdings.csv",
        help="Path to holdings CSV (default: portfolio_data/holdings.csv)",
    )
    p.add_argument(
        "--returns",
        default="sleeve_ERVol_Base_Nominal.csv",
        help="Path to sleeve-level ER/Vol (narrow) or wide monthly returns (default shown).",
    )
    p.add_argument(
        "--frontier-out",
        default="allocations_frontier_Base_Nominal.csv",
        help="CSV path for efficient frontier table.",
    )
    p.add_argument(
        "--allocation-out",
        default="allocation_targetVol_8_Base_Nominal.csv",
        help="CSV path for chosen target allocation (max Sharpe or constrained).",
    )
    p.add_argument(
        "--target-vol",
        type=float,
        default=None,
        help="If provided (e.g. 0.08 for 8%%), solve for maximum return at this volatility.",
    )
    p.add_argument(
        "--exclude-sleeves",
        default="Illiquid_Automattic",
        help="Comma-separated sleeves to exclude from optimization/stats (default: Illiquid_Automattic).",
    )
    return p.parse_args()


def _must_exist(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {p}")
    return p


# -------------------------------
# Name normalization / aliases
# -------------------------------
_ALIAS = {
    "CASH": "Cash",
    "US CORE": "US_Core",
    "US_CORE": "US_Core",
    "US VALUE": "US_Value",
    "US_VALUE": "US_Value",
    "US GROWTH": "US_Growth",
    "US_GROWTH": "US_Growth",
    "US SMALLVALUE": "US_SmallValue",
    "US SMALL VALUE": "US_SmallValue",
    "US_SMALLVALUE": "US_SmallValue",
    "US_SMALL_VALUE": "US_SmallValue",
    "INTL DM": "Intl_DM",
    "INTL_DM": "Intl_DM",
    "IG CORE": "IG_Core",
    "IG_CORE": "IG_Core",
    "IG INTL HEDGED": "IG_Intl_Hedged",
    "IG_INTL_HEDGED": "IG_Intl_Hedged",
    "EM USD": "EM_USD",
    "EM_USD": "EM_USD",
    "TREASURIES": "Treasuries",
    "TIPS": "TIPS",
    "ENERGY": "Energy",
    "EM": "EM",
    "ILLQUID_AUTOMATTIC": "Illiquid_Automattic",
    "ILLIQUID_AUTOMATTIC": "Illiquid_Automattic",
    "AUTOMATTIC": "Illiquid_Automattic",
}

def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    t = s.strip()
    t = t.replace("-", " ").replace("/", " ")
    t = re.sub(r"\s+", " ", t).strip()
    key = t.upper()
    if key in _ALIAS:
        return _ALIAS[key]
    # Generic fallback: convert spaces to underscores, Title-case words that match canonical style
    if " " in t:
        t2 = "_".join(w.capitalize() if w.isalpha() else w for w in t.split())
    else:
        t2 = t
    return t2


# -------------------------------
# IO: robust holdings loader
# -------------------------------
def _norm_map(cols: list[str]) -> dict[str, str]:
    return {c.lower().strip(): c for c in cols}

def load_holdings(path: str | Path) -> pd.DataFrame:
    """
    Return DataFrame with columns: Sleeve, Value (floats).
    Accepts many possible schemas; computes Value when absent.
    """
    p = _must_exist(path)
    df = pd.read_csv(p)
    if df.empty:
        sys.exit("[ERROR] holdings CSV is empty.")

        # --- Sleeve normalization & inference ---

        SLEEVE_ALIASES = {
            "US CORE": "US_Core", "US-CORE": "US_Core", "US_CORE": "US_Core", "CORE_US": "US_Core",
            "CORE US": "US_Core",
            "US VALUE": "US_Value", "US-VALUE": "US_Value", "US_VALUE": "US_Value",
            "US GROWTH": "US_Growth", "US-GROWTH": "US_Growth", "US_GROWTH": "US_Growth",
            "US SMALL VALUE": "US_SmallValue", "US SMALLVALUE": "US_SmallValue", "US_SMALL_VALUE": "US_SmallValue",
            "US-SMALL-VALUE": "US_SmallValue",
            "INTL DM": "Intl_DM", "INTL_DM": "Intl_DM", "INTERNATIONAL DM": "Intl_DM",
            "INTL-DEVELOPED": "Intl_DM", "DEVELOPED EX-US": "Intl_DM",
            "EM USD": "EM_USD", "EM_USD": "EM_USD", "EM (USD)": "EM_USD",
            "EM": "EM", "EMERGING": "EM",
            "IG CORE": "IG_Core", "IG_CORE": "IG_Core", "INVESTMENT GRADE CORE": "IG_Core",
            "IG INTL HEDGED": "IG_Intl_Hedged", "IG_INTL_HEDGED": "IG_Intl_Hedged", "INTL IG HEDGED": "IG_Intl_Hedged",
            "TREASURIES": "Treasuries", "UST": "Treasuries", "US TREASURIES": "Treasuries",
            "CASH": "Cash", "TIPS": "TIPS", "ENERGY": "Energy",
            "ILLIQUID AUTOMATTIC": "Illiquid_Automattic", "ILLIQUID_AUTOMATTIC": "Illiquid_Automattic",
        }

        def normalize_sleeve(x) -> str:
            if x is None:
                return ""
            t = str(x).strip()
            t = t.replace("—", "-").replace("–", "-")
            upper = t.upper().replace("_", " ").replace("-", " ").strip()
            return SLEEVE_ALIASES.get(upper, t.replace(" ", "_"))

        # 1) exact symbol→sleeve map (extend as needed)
        SYMBOL_TO_SLEEVE = {
            # US core/value/growth/small/value
            "VTI": "US_Core", "ITOT": "US_Core", "SPY": "US_Core",
            "VTV": "US_Value", "IVE": "US_Value",
            "IVW": "US_Growth", "QQQ": "US_Growth",
            "VBR": "US_SmallValue", "IJS": "US_SmallValue",

            # Intl developed / Emerging
            "VEA": "Intl_DM", "IEFA": "Intl_DM",
            "VWO": "EM", "IEMG": "EM",
            "EMB": "EM_USD",

            # Bonds
            "AGG": "IG_Core", "BND": "IG_Core",
            "BNDX": "IG_Intl_Hedged",
            "IEF": "Treasuries", "TLT": "Treasuries", "SHY": "Treasuries",

            # Cash & tips & sector
            "BIL": "Cash", "SGOV": "Cash", "CASH": "Cash",
            "TIP": "TIPS", "SCHP": "TIPS",
            "XLE": "Energy",
        }

        # 2) fallback by Name keywords (broad, extend if needed)
        NAME_KEYWORDS = [
            ("US_SmallValue", ["SMALL", "SMALL-CAP", "SMALL CAP", "SMALLCAP", "SV", "SMALL VALUE"]),
            ("US_Value", ["VALUE"]),
            ("US_Growth", ["GROWTH"]),
            ("US_Core", ["TOTAL US", "TOTAL STOCK", "S&P", "US EQUITY", "US CORE", "TOTAL MARKET", "CORE US"]),
            ("Intl_DM", ["DEVELOPED", "EAFE", "INTL", "INTERNATIONAL", "EX-US"]),
            ("EM_USD", ["EM USD", "EM (USD)", "EM USD BOND"]),
            ("EM", ["EMERGING", "EM "]),
            ("IG_Intl_Hedged", ["INTL HEDGED", "INTERNATIONAL HEDGED", "BNDX"]),
            ("IG_Core", ["AGGREGATE", "CORE BOND", "INVESTMENT GRADE", "IG CORE", "BND", "AGG"]),
            ("Treasuries", ["UST", "TREASUR", "T-NOTE", "T-BOND", "T-BILL", "GOVERNMENT"]),
            ("TIPS", ["TIPS", "INFLATION"]),
            ("Cash", ["CASH", "TREASURY BILL", "T-BILL", "ULTRA SHORT", "SGOV", "BIL"]),
            ("Energy", ["ENERGY", "OIL", "XLE"]),
            ("Illiquid_Automattic", ["AUTOMATTIC", "ILLIQUID"]),
        ]

        def infer_sleeve(symbol: str, name: str) -> str:
            s = str(symbol or "").strip().upper()
            n = str(name or "").strip().upper()
            if s in SYMBOL_TO_SLEEVE:
                return SYMBOL_TO_SLEEVE[s]
            for sleeve, keys in NAME_KEYWORDS:
                if any(k in n for k in keys):
                    return sleeve
            return "US_Core"  # safe default

        # --- apply logic: keep existing sleeves, infer only where null/blank ---
        if "Sleeve" not in df.columns:
            df["Sleeve"] = ""

        df["Sleeve"] = df["Sleeve"].fillna("").astype(str)

        mask_blank = df["Sleeve"].str.strip().eq("")
        if mask_blank.any():
            df.loc[mask_blank, "Sleeve"] = [
                infer_sleeve(sym, nm)
                for sym, nm in zip(df.loc[mask_blank, "Symbol"], df.loc[mask_blank, "Name"])
            ]

        # normalize everything
        df["Sleeve"] = df["Sleeve"].apply(normalize_sleeve)

    L = _norm_map(list(df.columns))

    # Sleeve-like column
    sleeve_col = None
    for key in ("sleeve", "class", "bucket", "assetclass", "sleeve_name"):
        if key in L:
            sleeve_col = L[key]
            break
    if sleeve_col is None:
        for key in ("symbol", "ticker", "name"):
            if key in L:
                sleeve_col = L[key]
                break
    if sleeve_col is None:
        sys.exit("[ERROR] could not find a Sleeve/Symbol/Name column to aggregate by.")

    # Value or Quantity*Price
    value_col = None
    for key in ("value", "marketvalue", "currentvalue", "currvalue"):
        if key in L:
            value_col = L[key]
            break
    if value_col is None:
        q_col = next((L[k] for k in ("quantity", "shares", "position") if k in L), None)
        px_col = next((L[k] for k in ("price", "pricepershare", "currentprice", "lastprice") if k in L), None)
        if q_col is None or px_col is None:
            sys.exit("[ERROR] holdings must include Value or (Quantity and Price).")
        df["__Value__"] = pd.to_numeric(df[q_col], errors="coerce").fillna(0.0) * pd.to_numeric(df[px_col], errors="coerce").fillna(0.0)
        value_col = "__Value__"

    tmp = (
        df.groupby(sleeve_col, as_index=False)[value_col]
        .sum()
        .rename(columns={sleeve_col: "Sleeve", value_col: "Value"})
    )
    tmp["Sleeve"] = tmp["Sleeve"].apply(normalize_name)
    tmp["Value"] = pd.to_numeric(tmp["Value"], errors="coerce").fillna(0.0)
    tmp = tmp[tmp["Value"].abs() > 1e-12].copy()
    return tmp[["Sleeve", "Value"]]


# -------------------------------
# IO: returns / ER / covariance
# -------------------------------
def load_returns_or_er_vol(path: str | Path) -> tuple[pd.Index, pd.Series, pd.DataFrame]:
    """
    Accept either:
      1) Narrow: columns [Ticker|Sleeve], ER, Vol, optional correlation block
      2) Wide: rows=dates, columns=tickers (periodic returns)
    """
    p = _must_exist(path)
    raw = pd.read_csv(p)

    id_col = None
    for k in ("Ticker", "Sleeve", "Symbol", "Name"):
        if k in raw.columns:
            id_col = k
            break

    # Narrow ER/Vol
    if id_col and {"ER", "Vol"}.issubset(raw.columns):
        raw = raw.copy()
        raw[id_col] = raw[id_col].astype(str).apply(normalize_name)
        tickers = raw[id_col].astype(str)
        mu = raw["ER"].astype(float)
        vol = raw["Vol"].astype(float)

        # Try correlation block
        num_cols = [c for c in raw.columns if c not in {id_col, "ER", "Vol"} and pd.api.types.is_numeric_dtype(raw[c])]
        Corr = None
        if num_cols:
            maybe = raw[num_cols].astype(float)
            if maybe.shape[1] == len(tickers) and np.all((maybe.values >= -1) & (maybe.values <= 1)):
                Corr = maybe.values
        if Corr is None:
            Corr = 0.20 * np.ones((len(tickers), len(tickers)))
            np.fill_diagonal(Corr, 1.0)

        D = np.diag(vol.values)
        cov = D @ Corr @ D
        mu.index = tickers
        cov = pd.DataFrame(cov, index=tickers, columns=tickers)
        return tickers, mu, cov

    # Wide periodic returns
    if id_col is None and any(pd.api.types.is_numeric_dtype(raw[c]) for c in raw.columns):
        R = raw.select_dtypes(include=[np.number]).copy()
        if R.shape[1] < 2:
            sys.exit("[ERROR] returns file has <2 numeric columns; cannot build covariance.")
        R.columns = [normalize_name(c) for c in R.columns]
        mu = R.mean(axis=0) * 12.0
        cov = R.cov() * 12.0
        tickers = mu.index.astype(str)
        cov.index = tickers
        cov.columns = tickers
        return tickers, mu, cov

    sys.exit("[ERROR] returns file not recognized. Provide ER/Vol (narrow) or a wide returns table.")


# -------------------------------
# Optimizers
# -------------------------------
def max_sharpe(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    n = len(mu)
    if _HAS_CVXPY:
        w = cp.Variable(n)
        ret = mu.values @ w
        risk = cp.quad_form(w, cov.values)
        lambdas = np.geomspace(1e-6, 10, 50)
        best = None
        best_s = -1e9
        for lam in lambdas:
            prob = cp.Problem(cp.Maximize(ret - lam * risk), [cp.sum(w) == 1, w >= 0])
            try:
                prob.solve(solver=cp.SCS, verbose=False, max_iters=8000)
            except Exception:
                try:
                    prob.solve(solver=cp.ECOS, verbose=False, max_iters=20000)
                except Exception:
                    continue
            if w.value is None:
                continue
            wv = np.clip(w.value, 0, None)
            if wv.sum() <= 0:
                continue
            wv = wv / wv.sum()
            r = float(mu.values @ wv)
            v = float(math.sqrt(max(0.0, wv @ cov.values @ wv)))
            s = r / (v + 1e-12)
            if s > best_s:
                best_s = s
                best = wv.copy()
        if best is None:
            raise RuntimeError("No feasible solution for max Sharpe.")
        return pd.Series(best, index=mu.index, name="Weight")

    # Heuristic fallback
    inv = np.linalg.pinv(cov.values)
    raw = inv @ mu.values
    raw = np.maximum(raw, 0)
    w = raw / raw.sum()
    return pd.Series(w, index=mu.index, name="Weight")


def max_return_at_vol(mu: pd.Series, cov: pd.DataFrame, target_vol: float) -> pd.Series:
    n = len(mu)
    if not _HAS_CVXPY:
        return max_sharpe(mu, cov)
    w = cp.Variable(n)
    ret = mu.values @ w
    risk = cp.quad_form(w, cov.values)
    cons = [cp.sum(w) == 1, w >= 0, cp.sqrt(risk) <= float(target_vol) + 1e-12]
    prob = cp.Problem(cp.Maximize(ret), cons)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=20000)
    if w.value is None:
        raise RuntimeError("No feasible solution at the requested volatility.")
    wv = np.clip(w.value, 0, None)
    wv = wv / wv.sum() if wv.sum() > 0 else wv
    return pd.Series(wv, index=mu.index, name="Weight")


# -------------------------------
# Stats & frontier
# -------------------------------
def portfolio_stats(w: pd.Series, mu: pd.Series, cov: pd.DataFrame) -> tuple[float, float, float]:
    r = float((mu * w).sum())
    v = float(math.sqrt(max(0.0, w.values @ cov.values @ w.values)))
    s = r / (v + 1e-12)
    return r, v, s


def save_frontier(mu: pd.Series, cov: pd.DataFrame, out_csv: str | Path):
    vols = np.linspace(0.02, max(0.30, float(np.sqrt(np.diag(cov.values)).max()) * 1.2), 21)
    rows = []
    for tv in vols:
        try:
            w = max_return_at_vol(mu, cov, tv)
            r, v, s = portfolio_stats(w, mu, cov)
            rows.append({"TargetVol": tv, "ER": r, "Vol": v, "Sharpe": s, **w.to_dict()})
        except Exception:
            continue
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()

    # Load inputs
    holdings = load_holdings(args.holdings)
    tickers, mu, cov = load_returns_or_er_vol(args.returns)

    # Exclusions
    excl = [normalize_name(s) for s in str(args.exclude_sleeves).split(",") if s.strip()]
    if excl:
        holdings = holdings[~holdings["Sleeve"].isin(excl)].copy()

    # Align names
    tickers_set = set([normalize_name(x) for x in tickers])
    holdings["Sleeve"] = holdings["Sleeve"].apply(normalize_name)

    overlap = sorted(set(holdings["Sleeve"]).intersection(tickers_set))
    if not overlap:
        # Diagnostic + graceful continue
        dbg = {
            "holdings_unique": sorted(set(holdings["Sleeve"])),
            "returns_unique": sorted(tickers_set),
        }
        Path("debug_no_overlap.json").write_text(pd.Series(dbg).to_json(indent=2))
        print("[WARN] After normalization, no sleeves overlap between holdings and returns.")
        print("       Wrote debugging info to debug_no_overlap.json")
        # Proceed with full returns universe to keep you unblocked
        pass
    else:
        # Restrict mu/cov to the overlap (recommended)
        mu = mu.loc[overlap]
        cov = cov.loc[overlap, overlap]

    # Solve
    if args.target_vol is not None:
        w = max_return_at_vol(mu, cov, float(args.target_vol))
    else:
        w = max_sharpe(mu, cov)

    er, vol, sharpe = portfolio_stats(w, mu, cov)

    # Report
    weights_pct = (w * 100).round(2).sort_values(ascending=False)
    print("Weights (%):")
    print(weights_pct)

    print(f"\nExpected Return %: {er*100:.2f}")
    print(f"Volatility %: {vol*100:.2f}")
    print(f"Sharpe: {sharpe:.2f}")

    # Correlation preview
    try:
        sd = np.sqrt(np.diag(cov.values))
        with np.errstate(divide="ignore", invalid="ignore"):
            Corr = cov.values / np.outer(sd, sd)
        Corr = np.clip(Corr, -1, 1)
        corr_df = pd.DataFrame(Corr, index=mu.index, columns=mu.index)
        print("\nCorrelation matrix:")
        print(corr_df.round(2))
    except Exception:
        pass

    # Outputs
    Path(args.frontier_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.allocation_out).parent.mkdir(parents=True, exist_ok=True)
    w.to_frame("Weight").to_csv(args.allocation_out)
    save_frontier(mu, cov, args.frontier_out)
    print(f"\nSaved: {args.allocation_out}")
    print(f"Saved: {args.frontier_out}")


if __name__ == "__main__":
    main()