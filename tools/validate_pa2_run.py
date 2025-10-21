#!/usr/bin/env python3
import argparse, subprocess, sys, re, json, shutil, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PY = ROOT.joinpath(".venv/bin/python")
PA2 = ROOT.joinpath("PA2.py")

def run_pa2(holdings:str, target_vol:float|None, returns_file:str|None)->str:
    if not PY.exists():
        raise SystemExit(f"Python virtualenv missing: {PY}")
    if not PA2.exists():
        raise SystemExit(f"PA2.py not found at: {PA2}")
    cmd = [str(PY), str(PA2), "--holdings", holdings]
    if returns_file:
        cmd += ["--returns-file", returns_file]
    if target_vol is not None:
        cmd += ["--target-vol", str(target_vol)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + ("\nSTDERR:\n"+(r.stderr or ""))
    if r.returncode != 0:
        print(out)
        raise SystemExit(r.returncode)
    return out

def parse_weights_from_stdout(text:str)->dict:
    lines = text.splitlines()
    # find "Weights (%):"
    try:
        i = next(k for k,l in enumerate(lines) if l.strip().lower().startswith("weights (%)"))
    except StopIteration:
        return {}
    weights = {}
    for l in lines[i+1:]:
        s = l.strip()
        if not s:
            break
        if s.lower().startswith(("expected return", "volatility", "sharpe", "mean period returns")):
            break
        # skip column header line like "Weight%" or "Sleeve    Weight%"
        if re.search(r'weight\%?$', s, re.I):
            continue
        m = re.match(r'^[\s]*([A-Za-z0-9_]+)\s+(-?\d+(?:\.\d+)?)\s*$', s)
        if not m:
            # allow pandas Series style: "US_Core          22.60"
            parts = s.split()
            if len(parts)>=2 and re.fullmatch(r'-?\d+(?:\.\d+)?', parts[-1]):
                name = " ".join(parts[:-1])
                val = float(parts[-1])
                weights[name] = val
            continue
        name, val = m.group(1), float(m.group(2))
        weights[name] = val
    return weights

def parse_metrics(text:str)->dict:
    # Expected Return %: 7.91
    # Volatility %: 8.00
    # Sharpe: 0.54
    m = {}
    er = re.search(r'Expected\s+Return\s*%\s*:\s*(-?\d+(?:\.\d+)?)', text, re.I)
    vo = re.search(r'Volatility\s*%\s*:\s*(-?\d+(?:\.\d+)?)', text, re.I)
    sh = re.search(r'Sharpe\s*:\s*(-?\d+(?:\.\d+)?)', text, re.I)
    if er: m["expected_return_pct"] = float(er.group(1))
    if vo: m["volatility_pct"] = float(vo.group(1))
    if sh: m["sharpe"] = float(sh.group(1))
    return m

def validate(weights:dict, metrics:dict, target_vol:float|None, tol_pct:float=0.5)->list[str]:
    issues=[]
    if not weights:
        issues.append("No weights parsed.")
        return issues
    total = sum(weights.values())
    if abs(total-100.0) > tol_pct:
        issues.append(f"Weights sum {total:.2f}% != 100% (±{tol_pct}%)")
    if target_vol is not None:
        v = metrics.get("volatility_pct")
        if v is None:
            issues.append("Volatility % not found in output.")
        else:
            tv = target_vol*100.0
            if abs(v - tv) > 0.5:
                issues.append(f"Volatility {v:.2f}% not within ±0.50% of target {tv:.2f}%")
    return issues

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdings", required=True)
    ap.add_argument("--target-vol", type=float, required=True)
    ap.add_argument("--returns-file")
    ap.add_argument("--dump-json", action="store_true")
    args = ap.parse_args()

    out = run_pa2(args.holdings, args.target_vol, args.returns_file)
    weights = parse_weights_from_stdout(out)
    metrics = parse_metrics(out)

    if args.dump_json:
        payload = {"weights": weights, "metrics": metrics, "stdout": out}
        pathlib.Path("validate_debug.json").write_text(json.dumps(payload, indent=2))

    issues = validate(weights, metrics, args.target_vol)
    if issues:
        print("[FAIL]")
        for s in issues: print(" -", s)
        raise SystemExit(1)

    print("[PASS] Weights parsed and validated.")
    print("Weights (%):", json.dumps(weights, indent=2, sort_keys=True))
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()