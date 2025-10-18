# portfolio_analysis.py
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# ------------- LOAD HOLDINGS -------------
df = pd.read_csv("holdings.csv")
df['Value'] = df['Value'].astype(float)

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
    return pd.Series({'Category':cat,'Geography':geo})

df = pd.concat([df, df.apply(classify,  axis=1)], axis=1)

# Collapse to sleeves (you can adjust)
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
# Map any unmapped ticker by simple rules
def map_symbol(sym, name):
    s = sym.upper(); n = name.upper()
    if s in map_to_proxy: return map_to_proxy[s]
    if 'UST' in n or 'TREAS' in n or 'STRIP' in n: return 'Treasuries'
    if 'INFLATION' in n: return 'TIPS'
    return 'US_Core'

df['Sleeve'] = [map_symbol(s, n) for s,n in zip(df['Symbol'], df['Name'])]

w = df.groupby('Sleeve')['Value'].sum()
w = w / w.sum()

# ------------- DOWNLOAD RETURNS -------------
import yfinance as yf
import re

def is_yf_symbol(s):
    # Accept letters, digits, dot or hyphen (for ETFs/ADRs), and reject obvious CUSIPs & cash
    if s.upper() in {"SPAXX","VMFXX","FDRXX","912810TP#"}:
        return False
    if re.fullmatch(r"[0-9]{6,}", s):     # CUSIPs/STRIPS-like
        return False
    return True

def load_prices(tickers, start="2015-01-01"):
    """
    Returns a clean price matrix (Date index, columns=tickers) using auto-adjusted CLOSE.
    Skips tickers that return no data.
    """
    # yfinance quirks:
    # - auto_adjust=True => 'Close' is already adjusted; no 'Adj Close' column exists.
    # - MultiIndex columns when multiple tickers; single Index for one.
    raw = yf.download(
        tickers=list(tickers),      # iterable ok
        start=start,
        auto_adjust=True,           # adjusted close embedded in 'Close'
        progress=False,
        group_by="ticker",          # force ticker-first MultiIndex if multi
        threads=True
    )

    # Normalize to DataFrame of CLOSE prices only
    if isinstance(raw.columns, pd.MultiIndex):
        # Select level ('Close') across all tickers
        close = raw.loc[:, (slice(None), "Close")]
        # Flatten columns to ticker symbols
        close.columns = close.columns.get_level_values(0)
    else:
        # Single ticker case: raw has columns like ['Open','High',...,'Close','Volume']
        if "Close" not in raw.columns:
            raise ValueError("Downloaded frame has no 'Close' column. Check tickers.")
        close = raw["Close"].to_frame()
        close.columns = list(tickers)[:1]  # name the single column with the ticker

    # Drop all-empty columns (bad/unsupported tickers)
    close = close.dropna(axis=1, how="all")

    # Optional: forward-fill occasional missing days, then drop leading NaNs
    close = close.ffill().dropna(how="any", axis=0)

    if close.shape[1] == 0:
        raise ValueError("No valid price columns returned. Verify ticker list.")
    return close



proxy_tickers = {
    'US_Core':'SCHB','US_Value':'VTV','US_SmallValue':'VBR','US_Growth':'IVW',
    'Intl_DM':'VXUS','EM':'VWO','Energy':'XLE',
    'IG_Core':'AGG','Treasuries':'IEF','TIPS':'TIP','EM_USD':'VWOB','IG_Intl_Hedged':'BNDX','Cash':'BIL'
}


tickers = list({proxy_tickers[k] for k in w.index if k in proxy_tickers})
tickers = [t for t in tickers if is_yf_symbol(t)]

px = load_prices(tickers, start="2015-01-01")
rets = px.resample('ME').last().pct_change().dropna()

# Align sleeves to downloaded columns
name_from_ticker = {v:k for k,v in proxy_tickers.items()}
rets = rets.rename(columns=name_from_ticker)
common = [s for s in w.index if s in rets.columns]
w = w[common]; w /= w.sum()
rets = rets[common]

mu = rets.mean()*12
cov = rets.cov()*12
rf = 0.035
port_mu = float(mu @ w)
port_sig = float(np.sqrt(w @ cov @ w))
sharpe = (port_mu - rf)/port_sig

print("Weights (%):"); print((100*w).round(2))
print("\nExpected Return %:", round(100*port_mu,2))
print("Volatility %:", round(100*port_sig,2))
print("Sharpe:", round(sharpe,2))
print("\nCorrelation matrix:"); print(rets.corr().round(2))

# ------------- EFFICIENT FRONTIER (cvxopt) -------------
try:
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    N = len(mu); S = matrix(cov.values); pbar = matrix(mu.values)
    G = matrix(-np.identity(N)); h = matrix(0.0, (N,1))
    A = matrix(1.0, (1,N)); b = matrix(1.0)
    target_rets = np.linspace(mu.min(), mu.max(), 60)
    vols, rets_list = [], []
    for m in target_rets:
        # add equality constraint for expected return
        A2 = matrix(np.vstack([np.ones(N), mu.values]))
        b2 = matrix([1.0, float(m)])
        sol = solvers.qp(S, matrix(np.zeros(N)), G, h, A2, b2)
        w_ = np.array(sol['x']).reshape(-1)
        vols.append(np.sqrt(w_ @ cov.values @ w_))
        rets_list.append(m)
    plt.figure(figsize=(6.0,4.2))
    plt.plot(np.array(vols)*100, np.array(rets_list)*100, color='navy', lw=2, label='Frontier')
    plt.scatter([port_sig*100],[port_mu*100], c='crimson', label='Current')
    plt.xlabel('Volatility (%)'); plt.ylabel('Expected Return (%)'); plt.title('Efficient Frontier (Sleeves)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("frontier.png", dpi=150)
    plt.show()
except Exception as e:
    print("\n[Frontier skipped] Install cvxopt or use cvxpy. Reason:", e)

    import numpy as np, pandas as pd, cvxpy as cp

    # rets: monthly return matrix; px already built
    mu = rets.mean() * 12
    Sigma = rets.cov() * 12
    tickers = mu.index.tolist()
    n = len(tickers)
    rf = 0.035

    # Current weights vector aligned to tickers (sum to 1)
    w_cur = pd.Series(weights_by_ticker, index=tickers).reindex(tickers).fillna(0).values


    def stats(w):
        er = float(mu.values @ w)
        vol = float((w @ Sigma.values @ w) ** 0.5)
        sh = (er - rf) / vol if vol > 0 else np.nan
        return er, vol, sh


    # Max-Sharpe (long-only)
    w = cp.Variable(n)
    excess = mu.values - rf
    prob = cp.Problem(cp.Maximize(excess @ w / cp.sqrt(cp.quad_form(w, Sigma.values) + 1e-12)),
                      [cp.sum(w) == 1, w >= 0])
    prob.solve(solver=cp.ECOS, max_iters=20000, abstol=1e-8, reltol=1e-8, feastol=1e-8)
    w_ms = np.array(w.value).flatten()

    # Min-Vol (long-only)
    w2 = cp.Variable(n)
    prob2 = cp.Problem(cp.Minimize(cp.quad_form(w2, Sigma.values)),
                       [cp.sum(w2) == 1, w2 >= 0])
    prob2.solve(solver=cp.ECOS, max_iters=20000, abstol=1e-8, reltol=1e-8, feastol=1e-8)
    w_mv = np.array(w2.value).flatten()

    # Report
    mu_c, vol_c, sh_c = stats(w_cur)
    mu_ms, vol_ms, sh_ms = stats(w_ms)
    mu_mv, vol_mv, sh_mv = stats(w_mv)

    print("\nEfficient Frontier (annualized)")
    print(f"Current   : ER={mu_c * 100:.2f}%, Vol={vol_c * 100:.2f}%, Sharpe={sh_c:.2f}")
    print(f"MaxSharpe : ER={mu_ms * 100:.2f}%, Vol={vol_ms * 100:.2f}%, Sharpe={sh_ms:.2f}")
    print(f"MinVol    : ER={mu_mv * 100:.2f}%, Vol={vol_mv * 100:.2f}%, Sharpe={sh_mv:.2f}")

    (pd.DataFrame({
        "Current": w_cur, "MaxSharpe": w_ms, "MinVol": w_mv
    }, index=tickers)
     .to_csv("allocations_frontier.csv"))