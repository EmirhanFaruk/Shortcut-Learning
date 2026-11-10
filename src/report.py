"""
report.py
---------
Reads every results.json under outputs/ and writes a single human-readable
REPORT.md that compares the runs, runs hypothesis tests, and states plain-English
conclusions -- so you don't have to open each file.

    python3 src/report.py                 # scans outputs/, writes outputs/REPORT.md
    python3 src/report.py --out outputs

Hypothesis testing
------------------
For each shortcut run we test:
    H0: the model does NOT rely on the cue  (accuracy is the same with the cue
        aligned to the label as with no cue at all).
    H1: the model relies on the cue (aligned accuracy > clean accuracy).
If per-sample predictions were saved (newer runs), we use McNemar's paired test
on the same test images. Otherwise we fall back to a two-proportion z-test using
the test-set size counted from data/. The verdict thresholds are stated inline.
"""
import argparse
import glob
import json
import os
from pathlib import Path

from scipy.stats import chi2, norm


# ---------- stats ----------
def mcnemar(y_pred_a, y_pred_b, y_true):
    """Paired test: is accuracy under regime A different from regime B?
    b = A-correct & B-wrong, c = A-wrong & B-correct."""
    b = c = 0
    for pa, pb, t in zip(y_pred_a, y_pred_b, y_true):
        a_ok, b_ok = (pa == t), (pb == t)
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1
    if b + c == 0:
        return 0.0, 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)        # continuity-corrected
    return stat, float(chi2.sf(stat, 1)), b, c


def two_prop_z(p1, p2, n):
    """Unpaired two-proportion z-test (conservative stand-in when we lack
    per-sample data; the paired McNemar test would be more powerful)."""
    if not n:
        return None, None
    pool = (p1 + p2) / 2
    se = (pool * (1 - pool) * (2 / n)) ** 0.5
    if se == 0:
        return None, None
    z = (p1 - p2) / se
    return z, float(norm.sf(abs(z)) * 2)


def count_test_images(data_dir):
    base = Path(data_dir) / "test"
    if not base.exists():
        return None
    n = 0
    for cls in ("benign", "malignant"):
        d = base / cls
        if d.exists():
            n += sum(1 for _ in d.glob("*"))
    return n or None


# ---------- loading ----------
def load_all(out_root):
    runs = []
    for rp in glob.glob(os.path.join(out_root, "**", "results.json"), recursive=True):
        try:
            with open(rp) as f:
                d = json.load(f)
            d["_path"] = rp
            runs.append(d)
        except Exception:
            pass
    return runs


def bucket(runs):
    b = {"phase1": [], "phase2": [], "sweep": [], "mitigation": {}}
    for d in runs:
        path = d["_path"].replace("\\", "/")
        cfg = d.get("config", {})
        if cfg.get("synthetic"):
            continue                       # ignore smoke runs
        if "/sweep/" in path:
            b["sweep"].append(d)
        elif "/mitigation/" in path:
            name = "mitigated" if cfg.get("mitigate") else "baseline"
            b["mitigation"][name] = d
        elif cfg.get("phase") == "clean":
            b["phase1"].append(d)
        elif cfg.get("phase") == "shortcut":
            b["phase2"].append(d)
    return b


def shortcut_test(d, data_dir):
    """Return (method, stat_str, pvalue, verdict) for one shortcut run."""
    preds = d.get("preds")
    if preds and "aligned" in preds and "none" in preds:
        stat, p, b, c = mcnemar(preds["aligned"]["y_pred"], preds["none"]["y_pred"],
                                preds["none"]["y_true"])
        method = f"McNemar paired test (discordant pairs b={b}, c={c})"
        stat_str = f"chi2={stat:.1f}"
    else:
        reg = d["test_regimes"]
        n = count_test_images(data_dir)
        z, p = two_prop_z(reg["aligned"]["acc"], reg["none"]["acc"], n)
        if z is None:
            return ("none", "test-set size unknown -> no p-value", None, None)
        method = f"two-proportion z-test (n={n}, unpaired approximation)"
        stat_str = f"z={z:.1f}"
    if p is None:
        return ("none", "n/a", None, None)
    sig = "reject H0" if p < 0.05 else "cannot reject H0"
    return (method, stat_str, p, sig)


# ---------- report ----------
def fnum(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x)


def build(out_root):
    runs = load_all(out_root)
    b = bucket(runs)
    data_dir = "data/ham_binary"
    if runs:
        data_dir = runs[0].get("config", {}).get("data_dir", data_dir)

    L = ["# Results report", ""]
    conclusions = []

    # ---- Phase 1 ----
    if b["phase1"]:
        L += ["## Phase 1 - clean classifier comparison", "",
              "| Model | Accuracy | AUC | Sensitivity | Specificity |",
              "|-------|----------|-----|-------------|-------------|"]
        best = None
        for d in sorted(b["phase1"], key=lambda r: r["config"]["model"]):
            m = d["test_clean"]; mdl = d["config"]["model"]
            L.append(f"| {mdl} | {fnum(m['acc'])} | {fnum(m['auc'])} | "
                     f"{fnum(m['sensitivity'])} | {fnum(m['specificity'])} |")
            if best is None or m["auc"] > best[1]:
                best = (mdl, m["auc"])
        L.append("")
        conclusions.append(f"Best clean architecture by AUC: **{best[0]}** (AUC {fnum(best[1])}).")

    # ---- Phase 2 ----
    if b["phase2"]:
        L += ["## Phase 2 - shortcut injection", ""]
        for d in b["phase2"]:
            reg = d["test_regimes"]
            L += [f"rho = {d['config']['rho']}", "",
                  "| Regime | Accuracy | AUC | Sensitivity |",
                  "|--------|----------|-----|-------------|"]
            for mode in ["none", "aligned", "swapped"]:
                m = reg[mode]
                L.append(f"| {mode} | {fnum(m['acc'])} | {fnum(m['auc'])} | {fnum(m['sensitivity'])} |")
            method, stat_str, p, verdict = shortcut_test(d, data_dir)
            sens_none = reg["none"]["sensitivity"]; sens_al = reg["aligned"]["sensitivity"]
            if p is None:
                eff = "rejected" if d["swap_drop"] > 0.05 else "not rejected"
                hyp_line = (f"- Hypothesis test H0 (no cue reliance): no p-value "
                            f"(run with data/ present, or re-run to save per-sample predictions); "
                            f"on effect size (swap drop {fnum(d['swap_drop'])}) H0 is **{eff}**")
            else:
                hyp_line = (f"- Hypothesis test H0 (no cue reliance): {method}, {stat_str}, "
                            f"p={'<0.001' if p < 1e-3 else fnum(p)} -> **{verdict}**")
            L += ["",
                  f"- Reliance (aligned - clean): **{fnum(d['reliance'])}**",
                  f"- Swap drop (aligned - swapped): **{fnum(d['swap_drop'])}**",
                  f"- Malignant sensitivity: {fnum(sens_al)} with cue vs **{fnum(sens_none)}** without it",
                  hyp_line, ""]
            strong = d["swap_drop"] > 0.2
            conclusions.append(
                f"Phase 2: the model {'**relied heavily** on the cue' if strong else 'showed weak cue reliance'} "
                f"(swap drop {fnum(d['swap_drop'])}); without the cue, malignant sensitivity "
                f"{'collapses to ' + fnum(sens_none) if sens_none < 0.2 else 'is ' + fnum(sens_none)}.")

    # ---- Sweep ----
    if b["sweep"]:
        rows = sorted(b["sweep"], key=lambda r: r["config"]["rho"])
        L += ["## Step 3 - adoption phase diagram (vary rho)", "",
              "| rho | reliance | swap drop |", "|-----|----------|-----------|"]
        for d in rows:
            L.append(f"| {d['config']['rho']} | {fnum(d['reliance'])} | {fnum(d['swap_drop'])} |")
        L.append("")
        sd = [d["swap_drop"] for d in rows]
        monotonic = all(sd[i] <= sd[i + 1] + 0.03 for i in range(len(sd) - 1))
        knee = next((d["config"]["rho"] for d in rows if d["swap_drop"] > 0.1), None)
        conclusions.append(
            "Step 3: shortcut dependence "
            + ("rises monotonically with cue predictiveness" if monotonic else "varies non-monotonically with cue predictiveness")
            + (f", taking off around rho={knee}" if knee else "") + ".")

    # ---- Mitigation ----
    if b["mitigation"].get("baseline") and b["mitigation"].get("mitigated"):
        base, mit = b["mitigation"]["baseline"], b["mitigation"]["mitigated"]
        bn, mn = base["test_regimes"]["none"], mit["test_regimes"]["none"]
        L += ["## Step 4 - mitigation (cue-randomizing augmentation)", "",
              "| Metric | Baseline | Mitigated |", "|--------|----------|-----------|",
              f"| Reliance | {fnum(base['reliance'])} | {fnum(mit['reliance'])} |",
              f"| Swap drop | {fnum(base['swap_drop'])} | {fnum(mit['swap_drop'])} |",
              f"| Clean-test accuracy | {fnum(bn['acc'])} | {fnum(mn['acc'])} |",
              f"| Clean-test sensitivity | {fnum(bn['sensitivity'])} | {fnum(mn['sensitivity'])} |", ""]
        worked = mit["swap_drop"] < 0.05 and base["swap_drop"] > 0.2
        conclusions.append(
            f"Step 4: mitigation {'**worked**' if worked else 'had limited effect'} - swap drop "
            f"{fnum(base['swap_drop'])} -> {fnum(mit['swap_drop'])}, and clean-test malignant "
            f"sensitivity {fnum(bn['sensitivity'])} -> {fnum(mn['sensitivity'])}.")

    # ---- Conclusions ----
    L = L[:2] + ["## Conclusions", ""] + [f"- {c}" for c in conclusions] + [""] + L[2:]
    return "\n".join(L) + "\n", conclusions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs", help="outputs root to scan")
    args = ap.parse_args()
    report, conclusions = build(args.out)
    path = os.path.join(args.out, "REPORT.md")
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w") as f:
        f.write(report)
    print("\n".join(f"- {c}" for c in conclusions) if conclusions
          else "No (non-smoke) results found under " + args.out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
