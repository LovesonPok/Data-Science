#!/usr/bin/env python3
"""
Descriptive Stats (Investigation A)
Keeping everything in PNG as we can use in pptx easyyy
- Overall descriptive table (PNG)
- Grouped descriptive tables (PNG) for slide:
    * bat_landing_number by rat_present (0/1)
    * bat_landing_to_food by risk (0/1)
    * seconds_after_rat_arrival by risk (0/1)
- Histograms for numeric variables
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from pathlib import Path

# Paths
DATA = Path("/Users/lovesonpokhrel/Documents/Data Science/cleaned_dataset.csv")
OUTDIR = Path("/Users/lovesonpokhrel/Documents/Data Science/figs_descriptive")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA)

# Variables
numeric_vars = [
    "bat_landing_to_food",
    "bat_landing_number",
    "food_availability",
    "seconds_after_rat_arrival",
]

# Helpers
def descriptive_table(series: pd.Series) -> dict:
    """Return compact descriptive stats for one numeric variable (sample-based)."""
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
        "outlier_cutoff": f">{round(hi, 2)}",
        "skewness": round(skew(s), 2),
    }

def save_table_png(df_table: pd.DataFrame, title: str, filename: Path):
    """Render a DataFrame as a PNG table (for slides)."""
    fig_h = 2 + 0.45 * max(1, len(df_table))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=df_table.round(2).values,
        colLabels=df_table.columns,
        rowLabels=df_table.index,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.2)
    plt.title(title, fontsize=12, pad=20)
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Table saved → {filename}")

def grouped_summary(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Mean / std / var / count by group (drops NaNs in value_col only)."""
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df[value_col], errors="coerce")
    tmp = df[[group_col]].copy()
    tmp[value_col] = s
    g = (
        tmp.dropna(subset=[value_col])
        .groupby(group_col)[value_col]
        .agg(mean="mean", std="std", variance="var", count="count")
    )
    # Make group labels look nice in row index
    g.index = [f"{group_col}={int(i)}" if pd.notna(i) else f"{group_col}=NaN" for i in g.index]
    return g

# everything in png to use in pptx
# Overall descriptive table
summary = {}
for col in numeric_vars:
    if col in df.columns:
        summary[col] = descriptive_table(df[col])

summary_df = pd.DataFrame(summary).T
save_table_png(summary_df, "Descriptive Statistics (Investigation A)", OUTDIR / "descriptive_summary.png")

# Grouped tables for slides
# 1) Bat landing number by rat_present
tbl1 = grouped_summary(df, "rat_present", "bat_landing_number")
if not tbl1.empty:
    save_table_png(tbl1, "Bat landing number by Rat presence", OUTDIR / "tbl_bat_landing_number_by_rat_present.png")

# 2) Bat landing to food by risk
tbl2 = grouped_summary(df, "risk", "bat_landing_to_food")
if not tbl2.empty:
    save_table_png(tbl2, "Bat landing to food by Risk", OUTDIR / "tbl_bat_landing_to_food_by_risk.png")

# 3) Seconds after rat arrival by risk
tbl3 = grouped_summary(df, "risk", "seconds_after_rat_arrival")
if not tbl3.empty:
    save_table_png(tbl3, "Seconds after rat arrival by Risk", OUTDIR / "tbl_seconds_after_rat_arrival_by_risk.png")

# Histograms
for col in numeric_vars:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=30, kde=False, color="skyblue")
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"{col}_hist.png", dpi=300)
        plt.close()
print(f"Histograms saved in {OUTDIR}")
