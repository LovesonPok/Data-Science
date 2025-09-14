#!/usr/bin/env python3
"""
Grouped Descriptive Stats (Investigation A)
- Save one PNG table
- Save one figure with 6 histograms (side by side)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from pathlib import Path

# ---------------- Paths ----------------
DATA = Path("/Users/lovesonpokhrel/Documents/Data Science/cleaned_dataset.csv")
OUTDIR = Path("/Users/lovesonpokhrel/Documents/Data Science/figs_descriptive")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------- Load dataset ----------------
df = pd.read_csv(DATA)

# ---------------- Helpers ----------------
def descriptive_table(series: pd.Series) -> dict:
    """Return compact descriptive stats for one numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    return {
        "count": int(len(s)),
        "mean": round(s.mean(), 2),
        "std": round(s.std(ddof=1), 2),
        "min": round(s.min(), 2),
        "25%": round(q1, 2),
        "median": round(s.median(), 2),
        "75%": round(q3, 2),
        "mode": (None if s.mode().empty else round(float(s.mode().iloc[0]), 2)),
        "max": round(s.max(), 2),
        "variance": round(s.var(ddof=1), 2),
        "outlier_cutoff": f">{round(hi, 2)}",
        "skewness": round(skew(s), 2),
    }

# ---------------- Define 6 groups ----------------
targets = [
    ("bat_landing_to_food", "risk", 0, "bat_landing_to_food (risk=0)"),
    ("bat_landing_to_food", "risk", 1, "bat_landing_to_food (risk=1)"),
    ("seconds_after_rat_arrival", "risk", 0, "seconds_after_rat_arrival (risk=0)"),
    ("seconds_after_rat_arrival", "risk", 1, "seconds_after_rat_arrival (risk=1)"),
    ("bat_landing_number", "rat_present", 0, "bat_landing_number (rat_present=0)"),
    ("bat_landing_number", "rat_present", 1, "bat_landing_number (rat_present=1)"),
]

# ---------------- Collect all summaries ----------------
all_summaries = {}
plot_data = []

for value_col, group_col, group_val, label in targets:
    if value_col in df.columns and group_col in df.columns:
        subset = df[df[group_col] == group_val][value_col]
        stats = descriptive_table(subset)
        if stats:
            all_summaries[label] = stats
            plot_data.append((subset, label, value_col))

summary_df = pd.DataFrame(all_summaries).T

# ---------------- Save single PNG table ----------------
fig_h = 2 + 0.45 * len(summary_df)
fig, ax = plt.subplots(figsize=(12, fig_h))
ax.axis("off")
tbl = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    rowLabels=summary_df.index,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.scale(1.2, 1.2)
plt.title("Descriptive Statistics by Group (Investigation A)", fontsize=12, pad=20)
outpath_table = OUTDIR / "descriptive_stats_grouped.png"
plt.savefig(outpath_table, bbox_inches="tight", dpi=300)
plt.close()
print(f"✅ Grouped descriptive table saved → {outpath_table}")

# ---------------- Save 6 histograms in one figure ----------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, (series, label, colname) in enumerate(plot_data):
    ax = axes[i]
    sns.histplot(series, bins=30, kde=False, color="skyblue", ax=ax)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(colname)
    ax.set_ylabel("Frequency")

# Hide any unused subplots (just in case)
for j in range(len(plot_data), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Histograms by Group (Investigation A)", fontsize=14, y=1.02)
plt.tight_layout()
outpath_hist = OUTDIR / "descriptive_histograms_grouped.png"
plt.savefig(outpath_hist, dpi=300, bbox_inches="tight")
plt.close()
print(f"Grouped histograms saved → {outpath_hist}")
