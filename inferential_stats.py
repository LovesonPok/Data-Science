# Inferential Statistics – three Welch one-sided tests (group 1 > group 0)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

DATA = Path("/Users/lovesonpokhrel/Documents/Data Science/cleaned_dataset.csv")
FIGDIR = DATA.parent / "figs_inferential"
FIGDIR.mkdir(parents=True, exist_ok=True)

def _boxplot_two_groups(series0, series1, value_label, group_col, figpath):
    plt.figure(figsize=(6,4))
    plt.boxplot([series0, series1],
                tick_labels=[f"{group_col}=0", f"{group_col}=1"],
                vert=True)
    plt.ylabel(value_label)
    plt.title(f"{value_label} by {group_col} (0 vs 1)")
    plt.tight_layout()
    plt.savefig(figpath, dpi=200, bbox_inches="tight")
    plt.close()

def run_two_sample_ttest(df, value_col, group_col, label0=0, label1=1, make_plot=True):
    """
    Welch’s two-sample t-test (one-sided: alternative='greater').

    H0: mean(value | group==label1) == mean(value | group==label0)
    H1: mean(value | group==label1)  >  mean(value | group==label0)
    """
    print("\n" + "="*72)
    print(f"[Two-sample t-test] {value_col} by {group_col} ({label0} vs {label1})")
    print("="*72)
    print("Hypotheses:")
    print(f"  H0: μ[{group_col}={label1}] = μ[{group_col}={label0}]")
    print(f"  H1: μ[{group_col}={label1}] > μ[{group_col}={label0}]  (one-sided)")

    sub = df[[value_col, group_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub[group_col] = pd.to_numeric(sub[group_col], errors="coerce")
    sub = sub.dropna()

    g0 = sub.loc[sub[group_col] == label0, value_col].astype(float)
    g1 = sub.loc[sub[group_col] == label1, value_col].astype(float)

    print(f"n[{group_col}={label0}] = {len(g0)} | mean = {g0.mean():.4f} | sd = {g0.std(ddof=1):.4f}")
    print(f"n[{group_col}={label1}] = {len(g1)} | mean = {g1.mean():.4f} | sd = {g1.std(ddof=1):.4f}")

    if len(g0) > 1 and len(g1) > 1:
        tstat, pval = ttest_ind(g1, g0, equal_var=False, nan_policy="omit",
                                alternative="greater")
        print(f"Welch t = {tstat:.4f} | one-sided p = {pval:.6f}")
        decision = "Reject H0 (evidence μ1 > μ0)" if pval < 0.05 else "Fail to reject H0"
        print("Decision @ α=0.05:", decision)
    else:
        print("Not enough observations in one or both groups to run the t-test.")
        tstat, pval = np.nan, np.nan

    if make_plot:
        figpath = FIGDIR / f"box_{value_col}_by_{group_col}.png"
        _boxplot_two_groups(g0.values, g1.values, value_col, group_col, figpath)
        print(f"Boxplot saved → {figpath}")

    return {
        "value": value_col, "group": group_col,
        "t": float(tstat) if np.isfinite(tstat) else np.nan,
        "p_one_sided": float(pval) if np.isfinite(pval) else np.nan,
        "n0": int(len(g0)), "n1": int(len(g1)),
        "mean0": float(g0.mean()) if len(g0) else np.nan,
        "mean1": float(g1.mean()) if len(g1) else np.nan
    }

def run_inferential_block():
    df = pd.read_csv(DATA)
    print(f"\nLoaded: {DATA} | rows={len(df)}")

    results = []

    # 1) Vigilance: bat_landing_to_food ~ risk (expect risk=1 > risk=0)
    if {"bat_landing_to_food", "risk"}.issubset(df.columns):
        results.append(run_two_sample_ttest(df, "bat_landing_to_food", "risk",
                                            label0=0, label1=1, make_plot=True))
    else:
        print("\nMissing columns for test: bat_landing_to_food ~ risk")

    # 2) Activity: bat_landing_number ~ rat_present (expect 1 > 0)
    if {"bat_landing_number", "rat_present"}.issubset(df.columns):
        results.append(run_two_sample_ttest(df, "bat_landing_number", "rat_present",
                                            label0=0, label1=1, make_plot=True))
    else:
        print("\nMissing columns for test: bat_landing_number ~ rat_present")

    # 3) Vigilance timing: seconds_after_rat_arrival ~ risk (expect 1 > 0)
    if {"seconds_after_rat_arrival", "risk"}.issubset(df.columns):
        results.append(run_two_sample_ttest(df, "seconds_after_rat_arrival", "risk",
                                            label0=0, label1=1, make_plot=True))
    else:
        print("\nMissing columns for test: seconds_after_rat_arrival ~ risk")

    if results:
        outcsv = FIGDIR / "ttest_summary.csv"
        pd.DataFrame(results).to_csv(outcsv, index=False)
        print(f"\nSummary saved → {outcsv}")

if __name__ == "__main__":
    run_inferential_block()
