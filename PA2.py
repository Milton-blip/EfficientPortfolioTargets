# portfolio_analysis.py
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import re, yfinance as yf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=str, default="Base",
                    choices=["Base","Reflation","HardLanding","Stagflation","Disinflation","Geopolitical"],
                    help="5Y macro scenario to adjust expected returns and volatility.")
parser.add_argument("--target_vol", type=float, default=0.08, help="Annualized volatility target (e.g., 0.08 for 8%%).")
parser.add_argument("--overlay_all", action="store_true",
                    help="Plot all macro scenarios on one frontier figure.")
parser.add_argument("--returns_basis", type=str, default="nominal",
                    choices=["nominal","real"],
                    help="Report/optimize in nominal or real terms.")
parser.add_argument("--inflation", type=float, default=None,
                    help="Override annual CPI inflation (e.g., 0.025). If not set, scenario defaults are used.")
args = parser.parse_args()

# -------------------- LOAD HOLDINGS --------------------
from pathlib import Path
import pandas as pd

base = Path(__file__).parent
df = pd.read_csv(base / "portfolio_data" / "holdings.csv")

def classify(row):
    s = str(row['Symbol']).upper()
    name = str(row['Name']).upper()
    debt_syms = {'AGG','SCHZ','FBND','BNDX','VWOB','SPAXX','FDRXX','VMFXX'}
    intl_eq_syms = {'VXUS','VWO','VPL','EMXC','FNDF','FNDE','FNDC'}
    if s in debt_syms or 'UST' in name or 'TREAS' in name or 'STRIP' in name or 'INFLATION' in name:
        cat = 'Debt'
    else:
        cat = 'Equity'
    geo = 'International' if (s in intl_eq_syms or 'ADR' in name or 'EMERGING' in name or 'PACIFIC' in name or 'INTERNATIONAL' in name) else 'Domestic'
    return pd.Series({'Category': cat, 'Geography': geo})

df = pd.concat([df, df.apply(classify, axis=1)], axis=1)

# -------------------- SLEEVE MAPPING --------------------
map_to_proxy = {
    'IVW':'US_Growth','VOOG':'US_Growth','AMZN':'US_Growth',
    'SCHB':'US_Core','DFAU':'US_Core','SCHM':'US_Core',
    'SCHA':'US_SmallValue','VBR':'US_SmallValue',
    'IUSV':'US_Value','VTV':'US_Value','VOOV':'US_Value','MGV':'US_Value',
    'VXUS':'Intl_DM','VPL':'Intl_DM','FNDF':'Intl_DM','FNDC':'Intl_DM',
    'VWO':'EM','EMXC':'EM','FNDE':'EM','TSM':'EM',
    'XLE':'Energy','VDE':'Energy',
    'AGG':'IG_Core','SCHZ':'IG_Core',
    'VWOB':'EM_USD','BNDX':'IG_Intl_Hedged',
    'SPAXX':'Cash','FDRXX':'Cash','VMFXX':'Cash'
}

def map_symbol(sym, name):
    s = str(sym).upper().strip()
    n = str(name).upper().strip()
    # Automattic → dedicated illiquid sleeve
    if s == "AUTOMATTIC" or "AUTOMATTIC" in n:
        return "Illiquid_Automattic"
    if s in map_to_proxy:
        return map_to_proxy[s]
    if 'UST' in n or 'TREAS' in n or 'STRIP' in n:
        return 'Treasuries'
    if 'INFLATION' in n:
        return 'TIPS'
    return 'US_Core'

df['Sleeve'] = [map_symbol(s, n) for s, n in zip(df['Symbol'], df['Name'])]

w = df.groupby('Sleeve')['Value'].sum()
w = w / w.sum()

# ---- Illiquid carve-out: keep Automattic weight fixed, optimize only on the rest ----
w_ill = float(w.get("Illiquid_Automattic", 0.0))
investable = [s for s in w.index if s != "Illiquid_Automattic"]
w_opt = (w[investable] / max(1e-12, (1.0 - w_ill))).copy()

# Helper to merge Automattic back later
def merge_back(weights_series):
    """
    weights_series: pd.Series over investable sleeves summing to 1
    returns pd.Series over investable + Illiquid_Automattic summing to 1
    """
    out = (1.0 - w_ill) * weights_series
    if w_ill > 0:
        out = out.copy()
        out.loc["Illiquid_Automattic"] = w_ill
    return out


# -------------------- DOWNLOAD RETURNS --------------------
def is_yf_symbol(s):
    if s.upper() in {"SPAXX","VMFXX","FDRXX","912810TP#"}: return False
    if re.fullmatch(r"[0-9]{6,}", s): return False
    return True

def load_prices(tickers, start="2015-01-01"):
    raw = yf.download(
        tickers=list(tickers),
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True
    )

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.loc[:, (slice(None), "Close")]
        close.columns = close.columns.get_level_values(0)
    else:
        if "Close" not in raw.columns:
            raise ValueError("No 'Close' column found.")
        close = raw["Close"].to_frame()
        close.columns = list(tickers)[:1]

    close = close.dropna(axis=1, how="all").ffill().dropna(how="any", axis=0)
    return close

proxy_tickers = {
    'US_Core':'SCHB','US_Value':'VTV','US_SmallValue':'VBR','US_Growth':'IVW',
    'Intl_DM':'VXUS','EM':'VWO','Energy':'XLE',
    'IG_Core':'AGG','Treasuries':'IEF','TIPS':'TIP',
    'EM_USD':'VWOB','IG_Intl_Hedged':'BNDX','Cash':'BIL'
}

tickers = list({proxy_tickers[k] for k in w.index if k in proxy_tickers})
tickers = [t for t in tickers if is_yf_symbol(t)]

px = load_prices(tickers, start="2015-01-01")
rets = px.resample('ME').last().pct_change().dropna()

name_from_ticker = {v: k for k, v in proxy_tickers.items()}
rets = rets.rename(columns=name_from_ticker)

# ----- investable universe (exclude Automattic sleeve) -----
investable = [s for s in rets.columns if s != "Illiquid_Automattic"]

# align weights: keep only investable sleeves, renormalize to 1 over investable piece
w_inv = (w_opt.reindex(investable).fillna(0.0)).copy()
w_inv /= max(1e-12, w_inv.sum())

# align returns to investable sleeves
rets_inv = rets[investable].copy()

# ----- Scenario-adjusted expected returns (mu_scn) and covariance scaling -----
# Baseline historical means
mu_hist = rets.mean() * 12  # annualized
# Per-sleeve add-ons to expected return (annual, absolute)
SCENARIO_MU_ADD = {
    "Base":         {"all": 0.00},
    "Disinflation": {"US_Growth": 0.01, "US_Core": 0.006, "US_Value": 0.004, "Intl_DM": 0.004,
                     "EM": 0.006, "IG_Core": 0.002, "Treasuries": 0.001, "TIPS": -0.002, "Energy": -0.005},
    "Reflation":    {"Energy": 0.030, "EM": 0.020, "TIPS": 0.015, "US_Value": 0.008, "US_Core": 0.004,
                     "US_Growth": 0.000, "IG_Core": -0.010, "Treasuries": -0.015},
    "HardLanding":  {"US_Growth": -0.030, "US_Core": -0.020, "US_Value": -0.020, "Intl_DM": -0.020,
                     "EM": -0.035, "Energy": -0.020, "IG_Core": 0.005, "Treasuries": 0.010, "TIPS": -0.005},
    "Stagflation":  {"US_Growth": -0.025, "US_Core": -0.015, "US_Value": -0.010, "Intl_DM": -0.010,
                     "EM": 0.005, "Energy": 0.020, "IG_Core": -0.015, "Treasuries": -0.020, "TIPS": 0.015},
    "Geopolitical": {"US_Growth": -0.015, "US_Core": -0.010, "US_Value": -0.008, "Intl_DM": -0.015,
                     "EM": -0.020, "Energy": 0.010, "IG_Core": 0.000, "Treasuries": 0.005, "TIPS": 0.006}
}
# Covariance (vol) multipliers by scenario (applied to Sigma AFTER shrinkage)
SCENARIO_VOL_MULT = {
    "Base": 1.00, "Disinflation": 0.95, "Reflation": 1.05, "HardLanding": 1.25, "Stagflation": 1.30, "Geopolitical": 1.20
}
# Annual CPI assumptions by scenario (used when --returns_basis=real)
SCENARIO_INFL = {
    "Base": 0.025,
    "Disinflation": 0.018,
    "Reflation": 0.035,
    "HardLanding": 0.020,
    "Stagflation": 0.045,
    "Geopolitical": 0.030,
}

# Build mu_scn by adding sleeve-specific adjustments (missing keys get 0)
mu_scn = mu_hist.copy()
adds = SCENARIO_MU_ADD.get(args.scenario, {"all": 0.00})
base_add = adds.get("all", 0.0)
mu_scn = mu_scn + base_add
for k, v in adds.items():
    if k == "all": continue
    if k in mu_scn.index:
        mu_scn[k] += v

# Optional clamp to avoid unrealistic inputs
mu_scn = mu_scn.clip(lower=-0.05, upper=0.18)

# Stash for downstream frontier code
MU_FOR_FRONTIER = mu_scn
VOL_MULT_FOR_FRONTIER = SCENARIO_VOL_MULT.get(args.scenario, 1.00)

# ----- Real vs Nominal handling -----
infl_used = args.inflation if args.inflation is not None else SCENARIO_INFL.get(args.scenario, 0.025)
if args.returns_basis == "real":
    MU_FOR_FRONTIER = MU_FOR_FRONTIER - infl_used
BASIS_LABEL = "Real" if args.returns_basis == "real" else "Nominal"

# ---- Risk-free handling (nominal vs real) ----
RF_NOMINAL = 0.035
RF_USED = RF_NOMINAL if args.returns_basis == "nominal" else ( (1+RF_NOMINAL)/(1+infl_used) - 1.0 )

mu = rets.mean() * 12
cov = rets.cov() * 12

# Align stats to sleeves that actually have return history
common_idx = w.index.intersection(mu.index)
if len(common_idx) < len(w.index):
    missing = [x for x in w.index if x not in mu.index]
    print("[NOTE] Excluding sleeves with no return history from stats:", missing)

mu = mu.reindex(common_idx)
cov = cov.reindex(index=common_idx, columns=common_idx)
w_stats = w.reindex(common_idx).fillna(0.0)

port_mu = float(mu @ w_stats)
port_sig = float(np.sqrt(w_stats @ cov.values @ w_stats))
sharpe = (port_mu - RF_USED) / port_sig if port_sig > 0 else float("nan")

print("Weights (%):"); print((100*w).round(2))
print(f"\nExpected Return %: {port_mu*100:.2f}")
print(f"Volatility %: {port_sig*100:.2f}")
print(f"Sharpe: {sharpe:.2f}")
print("\nCorrelation matrix:"); print(rets.corr().round(2))



# -------- Efficient Frontier (cvxpy) with realistic constraints --------
import numpy as np, pandas as pd, cvxpy as cp
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt


# === inputs for optimizer (investable only) ===
tickers = investable
n = len(tickers)

# scenario-adjusted expected returns, investable only
mu = MU_FOR_FRONTIER.reindex(tickers).astype(float)

# Ledoit–Wolf covariance on investable sleeves; apply scenario vol multiplier
from sklearn.covariance import LedoitWolf
Sigma = pd.DataFrame(LedoitWolf().fit(rets_inv.values).covariance_,
                     index=tickers, columns=tickers) * 12.0
Sigma *= (VOL_MULT_FOR_FRONTIER ** 2)
eps = 1e-8
Sigma_v = Sigma.values + eps*np.eye(n)

# current investable weights (sum to 1 over investable)
w_cur = w_inv.reindex(tickers).fillna(0.0).values

# bounds for investable sleeves only
ub = pd.Series(0.40, index=tickers)
lb = pd.Series(0.00, index=tickers)
if "Cash" in ub.index:
    ub["Cash"] = max(float(w.get("Cash", 0.0)), 0.02)
for s in ["Energy"]:
    if s in ub.index: ub[s] = 0.08
for s in ["US_Growth","US_Core","US_Value","US_SmallValue"]:
    if s in ub.index: ub[s] = 0.45
for s in ["IG_Core","Treasuries","TIPS","IG_Intl_Hedged","EM_USD"]:
    if s in ub.index:
        lb[s] = 0.00
        ub[s] = min(ub[s], 0.35)
if "EM" in ub.index:
    ub["EM"] = min(ub["EM"], 0.15)

LB = lb.reindex(tickers).fillna(0).values
UB = ub.reindex(tickers).fillna(0.40).values

def stats_investable(wts):
    er = float(mu.values @ wts)
    vol = float(np.sqrt(wts @ Sigma_v @ wts))
    sh = (er - RF_USED)/vol if vol > 0 else np.nan
    return er, vol, sh

# --- Max-Sharpe (SOC) on investable only ---
import cvxpy as cp
w_ms = cp.Variable(n)
s = cp.Variable()
cons_ms = [
    (mu.values - RF_USED) @ w_ms >= s,
    cp.quad_form(w_ms, Sigma_v) <= 1.0,
    cp.sum(w_ms) == 1.0,
    w_ms >= LB,
    w_ms <= UB
]
cp.Problem(cp.Maximize(s), cons_ms).solve(solver=cp.ECOS, max_iters=30000)

# --- Min-Vol on investable only ---
w_mv = cp.Variable(n)
cp.Problem(cp.Minimize(cp.quad_form(w_mv, Sigma_v)),
           [cp.sum(w_mv)==1.0, w_mv>=LB, w_mv<=UB]).solve(solver=cp.ECOS, max_iters=30000)

w_ms_v = np.array(w_ms.value).flatten()
w_mv_v = np.array(w_mv.value).flatten()

# ---- Automattic-aware helpers (investable-only optimize, then merge back) ----
ILLQ = "Illiquid_Automattic"   # must already exist in your lists if present
A_W = float(w.get(ILLQ, 0.0)) if ILLQ in w.index else 0.0

def merge_back(w_investable: pd.Series) -> pd.Series:
    """Return full-sleeve weights including Automattic fixed at A_W."""
    full = w_investable.reindex(tickers).fillna(0.0).copy()
    if ILLQ in rets.columns:
        # scale investable to (1 - A_W), set Automattic weight to A_W
        full *= (1.0 - A_W)
        full = full.reindex(rets.columns).fillna(0.0)
        full[ILLQ] = A_W
    return full

def report_stats(w_full: pd.Series):
    """Compute ER/Vol/Sharpe for full portfolio with Automattic assumed 0 ER/Vol."""
    # start from investable inputs
    mu_i = MU_FOR_FRONTIER.reindex(tickers).fillna(0.0).values
    Sig_i = Sigma_v  # investable covariance matrix (n x n)

    # split weights
    if ILLQ in w_full.index:
        A = float(w_full.get(ILLQ, 0.0))
    else:
        A = 0.0
    w_i = w_full.reindex(tickers).fillna(0.0).values
    # ER and Var for investable part
    er_i = float(mu_i @ w_i)
    var_i = float(w_i @ Sig_i @ w_i)
    # Automattic contribution assumed 0 ER and 0 Var -> no cross terms (conservative)
    er_full = er_i
    vol_full = var_i ** 0.5
    sh = (er_full - RF_USED) / vol_full if vol_full > 0 else np.nan
    return er_full, vol_full, sh

# Build "current / max-sharpe / min-vol" FULL portfolios (Automattic fixed)
w_cur_full = merge_back(w.reindex(tickers).fillna(0.0))
w_ms_full  = merge_back(pd.Series(w_ms_v, index=tickers))
w_mv_full  = merge_back(pd.Series(w_mv_v, index=tickers))

# 6) Smooth frontier curve (optimize investable only), then merge Automattic back
gammas = np.geomspace(1e-1, 5e2, 60)  # higher γ → lower risk
vols_full, ers_full = [], []

for g in gammas:
    w_g = cp.Variable(n)
    prob_g = cp.Problem(
        cp.Minimize(g*cp.quad_form(w_g, Sigma_v) - mu.values @ w_g),
        [cp.sum(w_g)==1.0, w_g>=LB, w_g<=UB]
    )
    prob_g.solve(solver=cp.ECOS, max_iters=20000)
    wv = np.array(w_g.value).flatten()
    w_full = merge_back(pd.Series(wv, index=tickers))
    er_f, vol_f, _ = report_stats(w_full)
    ers_full.append(er_f); vols_full.append(vol_f)

# 7) Report + plot using FULL stats (Automattic held constant)
mu_c, vol_c, sh_c = report_stats(w_cur_full)
mu_ms, vol_ms, sh_ms = report_stats(w_ms_full)
mu_mv, vol_mv, sh_mv = report_stats(w_mv_full)

print("\nEfficient Frontier (annualized, constrained; Automattic held constant):")
print(f"Current   : ER={mu_c*100:.2f}%, Vol={vol_c*100:.2f}%, Sharpe={sh_c:.2f}")
print(f"MaxSharpe : ER={mu_ms*100:.2f}%, Vol={vol_ms*100:.2f}%, Sharpe={sh_ms:.2f}")
print(f"MinVol    : ER={mu_mv*100:.2f}%, Vol={vol_mv*100:.2f}%, Sharpe={sh_mv:.2f}")

# Export full weights (includes Automattic as fixed)
alloc_full = pd.DataFrame({
    "Current":  w_cur_full,
    "MaxSharpe": w_ms_full,
    "MinVol":    w_mv_full
}).fillna(0.0)
alloc_full.to_csv(f"allocations_frontier_{args.scenario}_{BASIS_LABEL}.csv")

# Sleeve ER/Vol table (Automattic shown with 0/0 under our assumption)
stats_df = pd.DataFrame({
    "ER_%":  (MU_FOR_FRONTIER.reindex(alloc_full.index).fillna(0.0)*100).round(2),
    "Vol_%": 0.0
}, index=alloc_full.index)

# For investable sleeves, show their standalone vols from Sigma_v
standalone_vol = pd.Series(np.sqrt(np.diag(Sigma_v))*100, index=tickers)
stats_df.loc[standalone_vol.index, "Vol_%"] = standalone_vol.round(2)
# Ensure Automattic is 0
if ILLQ in stats_df.index:
    stats_df.loc[ILLQ, "Vol_%"] = 0.00

stats_df.to_csv(f"sleeve_ERVol_{args.scenario}_{BASIS_LABEL}.csv")

plt.figure(figsize=(6.8,4.4))
plt.plot(np.array(vols_full)*100, np.array(ers_full)*100, color="navy", lw=2, label="Frontier (Automattic fixed)")
plt.scatter(vol_c*100,  mu_c*100,  c="red",   label="Current",   zorder=3)
plt.scatter(vol_ms*100, mu_ms*100, c="green", label="MaxSharpe", zorder=3)
plt.scatter(vol_mv*100, mu_mv*100, c="blue",  label="MinVol",    zorder=3)
plt.xlabel("Volatility (%)"); plt.ylabel(f"Expected Return (%) — {BASIS_LABEL}")
plt.title(f"Efficient Frontier (Constrained, {args.scenario}, {BASIS_LABEL}; Automattic fixed)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"frontier_cvxpy_constrained_{args.scenario}_{BASIS_LABEL}.png", dpi=150)
plt.show()




# (plot using the Automattic-aware frontier arrays we built above)
plt.figure(figsize=(6.8,4.4))
plt.plot(np.array(vols_full)*100, np.array(ers_full)*100, color="navy", lw=2, label="Frontier (Automattic fixed)")
plt.scatter(vol_c*100,  mu_c*100,  c="red",   label="Current",   zorder=3)
plt.scatter(vol_ms*100, mu_ms*100, c="green", label="MaxSharpe", zorder=3)
plt.scatter(vol_mv*100, mu_mv*100, c="blue",  label="MinVol",    zorder=3)
plt.xlabel("Volatility (%)"); plt.ylabel(f"Expected Return (%) — {BASIS_LABEL}")
plt.title(f"Efficient Frontier (Constrained, {args.scenario}, {BASIS_LABEL}; Automattic fixed)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"frontier_cvxpy_constrained_{args.scenario}_{BASIS_LABEL}.png", dpi=150)
plt.show()

# Sleeve ER/Vol for sanity
stats_df = pd.DataFrame({
    "ER_%": (mu*100).round(2),
    "Vol_%": (np.sqrt(np.diag(Sigma_v))*100).round(2)
})
print("\nSleeve ER/Vol (annualized):")
print(stats_df)




# -------- Target Volatility Portfolio (8%) --------
target_vol = 0.08  # 8% annual volatility target

w_t = cp.Variable(n)
constraints_t = [
    cp.sum(w_t) == 1.0,
    w_t >= LB,
    w_t <= UB
]
prob_t = cp.Problem(cp.Maximize(mu.values @ w_t),
                    constraints_t + [cp.quad_form(w_t, Sigma_v) <= target_vol**2])
prob_t.solve(solver=cp.ECOS, max_iters=30000)

w_tgt = np.array(w_t.value).flatten()
er_tgt, vol_tgt, sh_tgt = stats_investable(w_tgt)

print("\nTarget Volatility Portfolio (8% annualized):")
print(f"Expected Return: {er_tgt*100:.2f}%")
print(f"Volatility:      {vol_tgt*100:.2f}%")
print(f"Sharpe Ratio:    {sh_tgt:.2f}")


plt.figure(figsize=(6.5,4.3))
plt.plot(np.array(vols_full)*100, np.array(ers_full)*100, color="navy", lw=1.5, label="Frontier (Automattic fixed)")
plt.scatter(vol_tgt*100, er_tgt*100, c="gold", s=100, label="Target 8% Vol", edgecolors="black")
plt.scatter(vol_c*100,  mu_c*100,  c="red",   label="Current",   zorder=3)
plt.scatter(vol_ms*100, mu_ms*100, c="green", label="MaxSharpe", zorder=3)
plt.scatter(vol_mv*100, mu_mv*100, c="blue",  label="MinVol",    zorder=3)
plt.xlabel("Volatility (%)"); plt.ylabel("Expected Return (%)")
plt.title(f"Efficient Frontier + {int(target_vol*100)}% Target ({args.scenario}, {BASIS_LABEL})")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(f"frontier_targetVol{int(target_vol*100)}_{args.scenario}_{BASIS_LABEL}.png", dpi=150)
pd.DataFrame({"Weight": w_tgt}, index=tickers).to_csv(f"allocation_targetVol_{int(target_vol*100)}_{args.scenario}_{BASIS_LABEL}.csv")
plt.show()


# ========= Overlay all scenarios on one plot =========
if args.overlay_all:
    import matplotlib.pyplot as plt
    import cvxpy as cp
    import numpy as np
    import pandas as pd
    try:
        from sklearn.covariance import LedoitWolf
        def shrink_cov(X):  # monthly -> annualized
            return pd.DataFrame(LedoitWolf().fit(X).covariance_,
                                index=rets.columns, columns=rets.columns) * 12.0
    except Exception:
        # Fallback diagonal-target shrinkage (no sklearn)
        def shrink_cov(X):
            S = np.cov(X.T, bias=False)          # monthly sample cov
            a = 0.25                              # shrinkage intensity
            F = np.diag(np.diag(S))
            Sig = ((1 - a) * S + a * F) * 12.0   # annualize
            return pd.DataFrame(Sig, index=rets.columns, columns=rets.columns)

    def build_mu_sigma(scn: str):
        mu_hist = rets.mean() * 12.0
        adds = SCENARIO_MU_ADD.get(scn, {"all": 0.0})
        base_add = adds.get("all", 0.0)
        mu = (mu_hist + base_add).copy()
        for k, v in adds.items():
            if k == "all": continue
            if k in mu.index: mu[k] += v
        mu = mu.clip(-0.05, 0.18)
        Sigma = shrink_cov(rets.values)
        mult = SCENARIO_VOL_MULT.get(scn, 1.0)
        Sigma *= (mult ** 2)
        Sig_v = Sigma.values + 1e-8 * np.eye(len(Sigma))
        return mu, Sigma, Sig_v

    tickers = rets.columns.tolist()
    n = len(tickers)
    LB = pd.Series(0.0, index=tickers)
    UB = pd.Series(0.40, index=tickers)
    if "Cash" in tickers: UB["Cash"] = max(float(w.get("Cash", 0.0)), 0.02)
    for s in ["Energy"]:
        if s in UB.index: UB[s] = 0.08
    for s in ["US_Growth","US_Core","US_Value","US_SmallValue"]:
        if s in UB.index: UB[s] = 0.45
    for s in ["IG_Core","Treasuries","TIPS","IG_Intl_Hedged","EM_USD"]:
        if s in UB.index: UB[s] = min(UB[s], 0.35)
    if "EM" in UB.index: UB["EM"] = min(UB.get("EM", 0.40), 0.15)
    LB = LB.values; UB = UB.values

    scenarios = ["Base","Disinflation","Reflation","HardLanding","Stagflation","Geopolitical"]
    colors = {"Base":"#1f77b4","Disinflation":"#2ca02c","Reflation":"#d62728",
              "HardLanding":"#9467bd","Stagflation":"#ff7f0e","Geopolitical":"#17becf"}

    plt.figure(figsize=(7.2,4.6))
    for scn in scenarios:
        mu_s, Sigma_s, Sig_v = build_mu_sigma(scn)
        # risk-aversion sweep → smooth frontier under bounds
        gammas = np.geomspace(1e-1, 5e2, 70)
        vols, ers = [], []
        for g in gammas:
            w_g = cp.Variable(n)
            cons = [cp.sum(w_g)==1.0, w_g >= LB, w_g <= UB]
            obj  = cp.Minimize(g*cp.quad_form(w_g, Sig_v) - mu_s.values @ w_g)
            cp.Problem(obj, cons).solve(solver=cp.ECOS, max_iters=20000)
            wv = np.array(w_g.value).flatten()
            er = float(mu_s.values @ wv)
            vol = float(np.sqrt(wv @ Sig_v @ wv))
            ers.append(er); vols.append(vol)
        plt.plot(np.array(vols)*100, np.array(ers)*100, color=colors[scn], lw=2, label=scn)

    plt.xlabel("Volatility (%)"); plt.ylabel("Expected Return (%)")
    plt.title(f"Efficient Frontiers by 5Y Macro Scenario ({BASIS_LABEL})")
    plt.legend(ncol=2, fontsize=9); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(f"frontier_overlay_all_scenarios_{BASIS_LABEL}.png", dpi=150)
    plt.show()