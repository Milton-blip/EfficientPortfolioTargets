#!/usr/bin/env python3
from __future__ import annotations

import argparse, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ANNUAL_MEAN_VOL = {
    "US_Core":            (0.07, 0.16),
    "US_Value":           (0.075, 0.17),
    "US_Growth":          (0.085, 0.20),
    "US_SmallValue":      (0.09,  0.24),
    "Intl_DM":            (0.065, 0.17),
    "EM":                 (0.085, 0.24),
    "IG_Core":            (0.035, 0.06),
    "Treasuries":         (0.02,  0.05),
    "Energy":             (0.06,  0.30),
    "Cash":               (0.015, 0.005),
    "IG_Intl_Hedged":     (0.03,  0.05),
    "Illiquid_Automattic":(0.00,  0.00),
}
FALLBACK_ANNUAL_MEAN = 0.05
FALLBACK_ANNUAL_VOL  = 0.18

GROUPS = {
    "equity":  {"US_Core","US_Value","US_Growth","US_SmallValue","Intl_DM","EM","Energy"},
    "bond":    {"IG_Core","Treasuries","IG_Intl_Hedged"},
    "cash":    {"Cash"},
    "other":   {"Illiquid_Automattic"},
}
BASE_RHO = {
    ("equity","equity"): 0.70,
    ("equity","bond"):   0.15,
    ("equity","cash"):   0.00,
    ("bond","bond"):     0.60,
    ("bond","cash"):     0.10,
    ("cash","cash"):     0.00,
}
DEFAULT_RHO = 0.20

def to_monthly(mu_a: float, vol_a: float) -> tuple[float,float]:
    mu_m = (1.0 + mu_a)**(1.0/12.0) - 1.0
    vol_m = vol_a / math.sqrt(12.0)
    return mu_m, vol_m

def group_for(sleeve: str) -> str:
    for g, members in GROUPS.items():
        if sleeve in members:
            return g
    return "equity"

def nearest_spd(a: np.ndarray) -> np.ndarray:
    b = (a + a.T) / 2.0
    vals, vecs = np.linalg.eigh(b)
    vals[vals < 1e-8] = 1e-8
    spd = (vecs * vals) @ vecs.T
    spd = (spd + spd.T) / 2.0
    return spd

def correlation_matrix(sleeves: list[str]) -> np.ndarray:
    n = len(sleeves)
    rho = np.eye(n, dtype=np.float32)
    for i in range(n):
        gi = group_for(sleeves[i])
        for j in range(i+1, n):
            gj = group_for(sleeves[j])
            key = (gi, gj) if (gi, gj) in BASE_RHO else (gj, gi)
            r = BASE_RHO.get(key, DEFAULT_RHO)
            rho[i, j] = r
            rho[j, i] = r
    try:
        np.linalg.cholesky(rho)
    except np.linalg.LinAlgError:
        rho = nearest_spd(rho.astype(np.float64)).astype(np.float32)
    return rho

def load_sleeves_from_map(p: Path) -> list[str]:
    df = pd.read_csv(p)
    if "Sleeve" not in df.columns:
        print(f"[ERROR] '{p}' must contain a 'Sleeve' column.", file=sys.stderr)
        sys.exit(1)
    sleeves = sorted({str(s) for s in df["Sleeve"].astype(str) if s and str(s).lower() != "nan"})
    if not sleeves:
        print(f"[ERROR] No sleeves found in {p}.", file=sys.stderr)
        sys.exit(1)
    return sleeves

def build_mean_vol_vectors(sleeves: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mus, vols = [], []
    for s in sleeves:
        mu_a, vol_a = ANNUAL_MEAN_VOL.get(s, (FALLBACK_ANNUAL_MEAN, FALLBACK_ANNUAL_VOL))
        mu_m, vol_m = to_monthly(mu_a, vol_a)
        mus.append(mu_m); vols.append(vol_m)
    return np.array(mus, dtype=np.float32), np.array(vols, dtype=np.float32)

def synthesize_returns(sleeves: list[str], months: int, seed: int|None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu, sig = build_mean_vol_vectors(sleeves)
    rho = correlation_matrix(sleeves)
    cov = (sig[:,None] * sig[None,:]) * rho
    try:
        L = np.linalg.cholesky(cov.astype(np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        cov = nearest_spd(cov.astype(np.float64)).astype(np.float32)
        L = np.linalg.cholesky(cov.astype(np.float64)).astype(np.float32)

    Z = rng.standard_normal(size=(months, len(sleeves))).astype(np.float32)
    X = (Z @ L.T) + mu[None, :]
    X = np.clip(X, -0.4, 0.4, out=X)
    return pd.DataFrame(X, columns=sleeves)

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic sleeve returns from sleeve_map.csv.")
    ap.add_argument("--sleeve-map", default="portfolio_data/sleeve_map.csv")
    ap.add_argument("--months", type=int, default=180)
    ap.add_argument("--start", default="2005-01")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="returns/sleeve_returns.csv")
    args = ap.parse_args()

    smp = Path(args.sleeve_map)
    outp = Path(args.out)
    if not smp.exists():
        print(f"[ERROR] Could not find sleeve map file at: {smp}", file=sys.stderr)
        sys.exit(1)

    sleeves = load_sleeves_from_map(smp)
    df = synthesize_returns(sleeves, args.months, args.seed)

    try:
        start_period = pd.Period(args.start, freq="M")
    except Exception:
        print(f"[ERROR] Invalid --start '{args.start}'. Use YYYY-MM.", file=sys.stderr)
        sys.exit(1)
    dates = [ (start_period + i).end_time.normalize() for i in range(args.months) ]
    df.insert(0, "Date", pd.to_datetime(dates))

    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    print(f"Wrote {outp} with sleeves: {', '.join(sleeves)}")

if __name__ == "__main__":
    main()