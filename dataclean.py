#!/usr/bin/env python3
"""
Notes from zoom meeting and conversation with Furdosa and others

dataclean.py — Investigation A cleaning (D1 & D2 separately, then 30-min window match and merge outer merge I believe)

Rules:
- D1: keep Investigation-A columns, parse start_time, drop 'habit' if blank OR has no alphabetic letters as numerical habits dont make
sense and dont give any meaning same with null or empty,
      standardize 'habit', set rat_present_d1 = 1 (all present) as there is always rat start and end.
- D2: keep Investigation-A columns, parse time, set rat_present_d2 = 1 iff (rat_minutes>0 or rat_arrival_number>0) basically means 
if present 1 if not 0.
- Merge: for EACH D1 event, attach the D2 row whose time <= start_time < time+30min (duplicate D2 onto D1 events) checks if the start_time 
falls inside the time of ds2.
- No imputation can be done as it alters the original expected outcome 
- IQR filter numeric columns (EXCLUDES rat_minutes & rat_arrival_number) iqr filter as per the lecture slide.
- Sort by time; save CSV; print diagnostics as both ds1 and ds2 are ascending order

Output: cleaned_dataset.csv
"""

from pathlib import Path
import pandas as pd
import re

#  my paths, # please change this if you want to run with no error
P1  = Path("/Users/lovesonpokhrel/Documents/Data Science/dataset1.csv")
P2  = Path("/Users/lovesonpokhrel/Documents/Data Science/dataset2.csv")
OUT = Path("/Users/lovesonpokhrel/Documents/Data Science/cleaned_dataset.csv")

# Helpers
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return out

def drop_investigation_b_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Investigation-B / seasonal columns (not used for A)."""
    b_cols = [
        "month", "season", "sunset_time", "sunset", "sunrise", "sunrise_time",
        "hours_after_sunset", "daylength", "dawn", "dusk"
    ]
    return df.drop(columns=[c for c in b_cols if c in df.columns], errors="ignore")

def iqr_limits(series: pd.Series):
    """Return lower/upper bounds via IQR rule."""
    s = pd.to_numeric(series, errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return lo, hi

# Habit standardisation (locked)
_CORE = {"bat", "rat", "pick"}

_CANON_SET_2 = {
    frozenset({"bat", "pick"}): "bat_and_pick",
    frozenset({"bat", "rat"}):  "bat_and_rat",
    frozenset({"pick", "rat"}): "pick_and_rat",
}
_CANON_SET_3 = {frozenset({"bat", "rat", "pick"}): "bat_and_rat_and_pick"}

# Exact/near-exact mappings to desired labels as discussed with teammates
_SYNONYM_MAP = {
    # rat attack family -> keep as rat_attack
    "rat attack": "rat_attack",
    "attack_rat": "rat_attack",

    # eating variants you want collapsed to eating_bat_pick
    "eating_and_bat_and_pick": "eating_bat_pick",

    # common cleanups / typos
    "not_sure_rat": "rat",
    "rat_and_rat": "rat",
    "bat_figiht": "bat_fight",
    "fight_bat": "bat_fight",
    "bats": "bat",

    # other / others / other_directions -> other
    "others": "other",
    "other directions": "other",
    "other_directions": "other",

    # keep-as-is families you explicitly want preserved
    "other_bat_rat": "other_bat_rat",
    "eating_bat_pick": "eating_bat_pick",
    "eating_bat_rat_pick": "eating_bat_rat_pick",

    # keep-as-is flags
    "fast_far": "fast_far",
    "bat_and_pick_far": "bat_and_pick_far",

    # typo fix
    "pup_and_mon": "pup_and_mum",
}

# Labels that must NOT be collapsed after mapping to be in safe side
_LOCKED_LABELS = {
    "rat_attack",
    "eating_bat_pick",
    "eating_bat_rat_pick",
    "other_bat_rat",
    "fast_far",
    "bat_and_pick_far",
    "pup_and_mon",
}

def _norm_token(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    return s

def _canonical_from_tokens(tokset: frozenset) -> str | None:
    if tokset in _CANON_SET_2: return _CANON_SET_2[tokset]
    if tokset in _CANON_SET_3: return _CANON_SET_3[tokset]
    return None

def canonicalize_habit(x):
    """
    Standardize to consolidated set:
      - normalize punctuation/spacing
      - map synonyms/typos to preferred labels
      - lock certain labels to prevent further collapsing
      - canonicalize {bat,rat,pick} combinations order-invariant
      - special-case bat_fight (+ optional rat)
    """
    if pd.isna(x): return None
    s = _norm_token(str(x))
    if not s: return None

    # direct mapping first (desired label)
    if s in _SYNONYM_MAP:
        s = _SYNONYM_MAP[s]
        if s in _LOCKED_LABELS:
            return s

    # preserve any 'other*' label exactly as normalised
    if "other" in s:
        return s

    parts = [p for p in s.split("_") if p]

    # special ‘bat fight’ pattern
    if "fight" in parts and "bat" in parts:
        return "bat_fight_and_rat" if "rat" in parts else "bat_fight"

    # order-invariant actor-set canonicalisation
    tokset = frozenset([p for p in parts if p in _CORE])
    canon = _canonical_from_tokens(tokset)
    if canon: return canon

    # singletons
    if tokset == frozenset({"bat"}):  return "bat"
    if tokset == frozenset({"rat"}):  return "rat"
    if tokset == frozenset({"pick"}): return "pick"

    # otherwise return the normalised string (fast, gaze, bowl_out, no_food, etc.)
    return s

# Load & standardise
df1 = normalize_cols(pd.read_csv(P1))  # dataset 1 (events)
df2 = normalize_cols(pd.read_csv(P2))  # dataset 2 (30-min windows)

# Keep ONLY Investigation-A columns
keep1 = ["start_time", "habit", "bat_landing_to_food", "risk", "reward", "seconds_after_rat_arrival"]
keep2 = ["time", "bat_landing_number", "rat_minutes", "rat_arrival_number", "food_availability"]
df1 = df1[[c for c in keep1 if c in df1.columns]].copy()
df2 = df2[[c for c in keep2 if c in df2.columns]].copy()

# Drop Investigation-B columns if any slipped through
df1 = drop_investigation_b_cols(df1)
df2 = drop_investigation_b_cols(df2)

# Parse times
df1["start_time"] = pd.to_datetime(df1["start_time"], errors="coerce", dayfirst=True)
df2["time"]       = pd.to_datetime(df2["time"],       errors="coerce", dayfirst=True)
df1 = df1[df1["start_time"].notna()].copy()
df2 = df2[df2["time"].notna()].copy()

# D1 cleaning: drop 'habit' that is blank or has no alphabet
d1_before = len(df1)
if "habit" in df1.columns:
    habit_str = df1["habit"].astype("string").str.strip()
    mask_blank = habit_str.isna() | (habit_str == "")
    mask_no_alpha = ~habit_str.str.contains(r"[A-Za-z]", na=True)
    drop_mask = mask_blank | mask_no_alpha
    dropped_blank = int(mask_blank.sum())
    dropped_noalpha = int(mask_no_alpha.sum())
    df1 = df1.loc[~drop_mask].copy()

    # standardize habit in-place
    uniq_before = df1["habit"].nunique(dropna=True)
    df1["habit"] = df1["habit"].apply(canonicalize_habit)
    uniq_after = df1["habit"].nunique(dropna=True)
    print(f"[DF1:habit] unique before: {uniq_before} -> after: {uniq_after}")
else:
    dropped_blank = dropped_noalpha = 0
print(f"[DF1] rows before: {d1_before} | dropped blank: {dropped_blank} | "
      f"dropped no-alpha habit: {dropped_noalpha} | after: {len(df1)}")

# Rat presence flags
df1["rat_present_d1"] = 1
df2["rat_minutes"]        = pd.to_numeric(df2.get("rat_minutes"), errors="coerce").fillna(0)
df2["rat_arrival_number"] = pd.to_numeric(df2.get("rat_arrival_number"), errors="coerce").fillna(0)
df2["rat_present_d2"]     = ((df2["rat_minutes"] > 0) | (df2["rat_arrival_number"] > 0)).astype(int)
print(f"[D2] rat_present_d2 counts (pre-merge): {df2['rat_present_d2'].value_counts(dropna=False).to_dict()}")

# 30-MIN WINDOW MATCH (duplicate D2 onto D1 events)
# Sort for merge_asof
df1_sorted = df1.sort_values("start_time").reset_index(drop=True)
df2_sorted = df2.sort_values("time").reset_index(drop=True)

# Left-asof: for each D1.start_time, find the most recent D2.time within 30 minutes
merged = pd.merge_asof(
    left=df1_sorted, right=df2_sorted,
    left_on="start_time", right_on="time",
    direction="backward", tolerance=pd.Timedelta(minutes=30),
    suffixes=("_d1", "_d2")
)

# Optional: also keep D2 rows that matched no D1 event
INCLUDE_LONELY_D2 = True
if INCLUDE_LONELY_D2:
    matched_times = merged["time"].dropna().unique()
    lonely_d2 = df2_sorted[~df2_sorted["time"].isin(matched_times)].copy()
    if not lonely_d2.empty:
        # bring to same schema as merged (create missing D1 cols as NaN)
        for col in df1_sorted.columns:
            if col not in lonely_d2.columns:
                lonely_d2[col] = pd.NA
        if "rat_present_d1" not in lonely_d2.columns:
            lonely_d2["rat_present_d1"] = 0
        merged = pd.concat([merged, lonely_d2[merged.columns]], ignore_index=True)

# Combine rat_present = OR(D1, D2)
for col in ["rat_present_d1", "rat_present_d2"]:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
    else:
        merged[col] = 0
merged["rat_present"] = merged[["rat_present_d1", "rat_present_d2"]].max(axis=1)

#  keep only final rat_present
merged = merged.drop(columns=["rat_present_d1", "rat_present_d2"])


# IQR outlier removal
# didnt IQR-filter rat_minutes or rat_arrival_number
num_cols = [c for c in ["bat_landing_number", "bat_landing_to_food", "food_availability"]
            if c in merged.columns]
for col in num_cols:
    lo, hi = iqr_limits(merged[col])
    before = len(merged)
    merged = merged[(merged[col].isna()) | ((merged[col] >= lo) & (merged[col] <= hi))]
    after = len(merged)
    print(f"[IQR] {col}: dropped {before - after} (bounds: {lo:.3f} .. {hi:.3f})")

# Column order, sort & save
front = [c for c in ["start_time", "time", "rat_present", "habit"] if c in merged.columns]
others = [c for c in merged.columns if c not in front]
merged = merged[front + others]

# Sort by event time (prefer D1 start_time; fall back to D2 time)
sort_key = merged["start_time"].combine_first(merged["time"])
merged = merged.iloc[sort_key.argsort(kind="mergesort")].reset_index(drop=True)

# Final diagnostics
print(f"[FINAL] rows: {len(merged)} | cols: {len(merged.columns)}")
print(f"[FINAL] rat_present counts: {merged['rat_present'].value_counts(dropna=False).to_dict()}")

merged.to_csv(OUT, index=False)
print("✅ Cleaned dataset saved.")
print(f"Saved to: {OUT}")
#completed here till futher coding, 