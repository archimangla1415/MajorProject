# Phase 2 — Analytics Engine (equity robust)
# Input:  /Users/nehal/Desktop/test/dataset_phase1_ready.xlsx
# Output: /Users/nehal/Desktop/test/phase2_analytics_with_ACT_equity.xlsx

import pandas as pd
import numpy as np
from pathlib import Path

pd.options.mode.chained_assignment = None

# ---------- CONFIG ----------
INPUT_PATH = "/Users/nehal/Desktop/test/dataset_phase1_ready.xlsx"
OUTPUT_PATH = "/Users/nehal/Desktop/test/phase2_analytics_with_ACT_equity.xlsx"

# Column mapping (from your dataset)
COL_ERW = "ERW_SAT"
COL_MATH = "Math_SAT"
COL_TOTAL = "Total_SAT"
COL_TSI = "Above_TSI_Both_Rate"
COL_PART = "Part_Rate"

COL_ACT_ENG = "English_ACT"
COL_ACT_MATH = "Math_ACT"

COL_CAMPUS = "Campus"
COL_DIST = "District"
COL_REGION = "Region"
COL_YEAR = "year"

# Weights (disparity removed by default here)
WEIGHTS = {
    "sat_erw": 0.22,
    "sat_math": 0.22,
    "act_eng": 0.11,
    "act_math": 0.11,
    "tsi": 0.17,
    "participation": 0.10,
    "decline": 0.07
}
_total_w = sum(WEIGHTS.values())
if _total_w != 0:
    WEIGHTS = {k: v/_total_w for k, v in WEIGHTS.items()}

# ---------- HELPERS ----------
def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    if s.isna().all():
        return pd.Series(0.0, index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)

def add_rolling_mean(df, group_col, col_name, out_col):
    if col_name in df.columns and group_col in df.columns:
        df[out_col] = df.groupby(group_col)[col_name].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    else:
        df[out_col] = np.nan
    return df

# ---------- LOAD & PREP ----------
print("Loading:", INPUT_PATH)
df = pd.read_excel(INPUT_PATH, engine="openpyxl")
print("Shape:", df.shape)

target_cols = [COL_ERW, COL_MATH, COL_TOTAL, COL_TSI, COL_PART, COL_ACT_ENG, COL_ACT_MATH]
for c in target_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ensure campus/year exist
if COL_CAMPUS not in df.columns:
    raise KeyError(f"Required column '{COL_CAMPUS}' not found in input.")
if COL_YEAR not in df.columns:
    raise KeyError(f"Required column '{COL_YEAR}' not found in input.")

df = df.dropna(subset=[COL_CAMPUS, COL_YEAR])
# safe cast year to int when possible
try:
    df[COL_YEAR] = df[COL_YEAR].astype(int)
except:
    df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors='coerce')
    df = df.dropna(subset=[COL_YEAR])
    df[COL_YEAR] = df[COL_YEAR].astype(int)

# Ensure District and Region columns exist to avoid merge KeyErrors later
if COL_DIST not in df.columns:
    df[COL_DIST] = np.nan
if COL_REGION not in df.columns:
    df[COL_REGION] = np.nan

df = df.sort_values([COL_CAMPUS, COL_YEAR]).reset_index(drop=True)
print("Data sorted and prepared.")

# ---------- PERCENTILE-BASED COMPONENTS (per year) ----------
print("Computing percentile ranks per year for SAT and ACT components...")
risk_mapping = {
    COL_ERW: "erw_risk_c",
    COL_MATH: "math_risk_c",
    COL_ACT_ENG: "act_eng_risk_c",
    COL_ACT_MATH: "act_math_risk_c",
    COL_TSI: "tsi_risk_c",
    COL_PART: "part_risk_c",
}

for col, risk_c_name in risk_mapping.items():
    if col in df.columns:
        pct_col = col.lower().replace("_", "") + "_pct"
        # handle years with few non-null values gracefully
        df[pct_col] = df.groupby(COL_YEAR)[col].transform(lambda s: s.rank(pct=True, method="average") if s.notna().any() else np.nan)
        df[risk_c_name] = 1.0 - df[pct_col]
    else:
        df[risk_c_name] = 0.0

# ---------- DECLINE COMPONENT (3-year rolling mean) ----------
print("Computing 3-year rolling means for decline metrics...")
df = df.sort_values([COL_CAMPUS, COL_YEAR]).reset_index(drop=True)

df = add_rolling_mean(df, COL_CAMPUS, COL_ERW, "erw_3yr_mean")
df = add_rolling_mean(df, COL_CAMPUS, COL_MATH, "math_3yr_mean")
df = add_rolling_mean(df, COL_CAMPUS, COL_ACT_ENG, "act_eng_3yr_mean")
df = add_rolling_mean(df, COL_CAMPUS, COL_ACT_MATH, "act_math_3yr_mean")
df = add_rolling_mean(df, COL_CAMPUS, COL_TSI, "tsi_3yr_mean")

df["erw_decline"] = (df.get("erw_3yr_mean", 0) - df.get(COL_ERW, 0)).clip(lower=0)
df["math_decline"] = (df.get("math_3yr_mean", 0) - df.get(COL_MATH, 0)).clip(lower=0)
df["act_eng_decline"] = (df.get("act_eng_3yr_mean", 0) - df.get(COL_ACT_ENG, 0)).clip(lower=0)
df["act_math_decline"] = (df.get("act_math_3yr_mean", 0) - df.get(COL_ACT_MATH, 0)).clip(lower=0)
df["tsi_decline"] = (df.get("tsi_3yr_mean", 0) - df.get(COL_TSI, 0)).clip(lower=0)

decline_cols = ["erw_decline", "math_decline", "act_eng_decline", "act_math_decline", "tsi_decline"]
decline_present = [c for c in decline_cols if c in df.columns]

if decline_present:
    df["decline_raw"] = df[decline_present].max(axis=1).fillna(0.0)
    df["decline_c"] = normalize_series(df["decline_raw"])
else:
    df["decline_c"] = 0.0

# ---------- FINAL RISK SCORE (compose components) ----------
print("Composing final risk score (0-100) WITHOUT disparity...")
df["risk_score"] = 0.0
risk_components = {
    "sat_erw": df.get("erw_risk_c", pd.Series(0.0, index=df.index)),
    "sat_math": df.get("math_risk_c", pd.Series(0.0, index=df.index)),
    "act_eng": df.get("act_eng_risk_c", pd.Series(0.0, index=df.index)),
    "act_math": df.get("act_math_risk_c", pd.Series(0.0, index=df.index)),
    "tsi": df.get("tsi_risk_c", pd.Series(0.0, index=df.index)),
    "participation": df.get("part_risk_c", pd.Series(0.0, index=df.index)),
    "decline": df.get("decline_c", pd.Series(0.0, index=df.index)),
}

for key, component_series in risk_components.items():
    if key in WEIGHTS:
        df["risk_score"] += WEIGHTS[key] * component_series.fillna(0.0)

df["risk_score"] = df["risk_score"].clip(0, 1) * 100.0

# ---------- ADD REQUESTED COLUMNS ----------
df["sat_risk_score"] = ((df.get("erw_risk_c", 0.0).fillna(0.0) + df.get("math_risk_c", 0.0).fillna(0.0)) / 2.0) * 100.0
df["act_risk_score"] = ((df.get("act_eng_risk_c", 0.0).fillna(0.0) + df.get("act_math_risk_c", 0.0).fillna(0.0)) / 2.0) * 100.0
df["tsi"] = df.get("tsi_risk_c", 0.0).fillna(0.0) * 100.0

# ---------- BENCHMARKS: Campus vs District vs Region ----------
metrics = [c for c in [COL_ERW, COL_MATH, COL_TOTAL, COL_TSI, COL_PART, COL_ACT_ENG, COL_ACT_MATH] if c in df.columns]

bench_rows = []
for year, sub in df.groupby(COL_YEAR):
    district_avg = sub.groupby([COL_DIST])[metrics].mean().reset_index().rename(columns={c: f"{c}_district_avg" for c in metrics})
    region_avg = sub.groupby([COL_REGION])[metrics].mean().reset_index().rename(columns={c: f"{c}_region_avg" for c in metrics})
    campus_avg = sub.groupby([COL_CAMPUS, COL_DIST, COL_REGION])[metrics].mean().reset_index()
    merged = campus_avg.merge(district_avg, on=COL_DIST, how="left").merge(region_avg, on=COL_REGION, how="left")
    for _, r in merged.iterrows():
        row = {"Campus": r[COL_CAMPUS], "Year": year, "District": r[COL_DIST], "Region": r[COL_REGION]}
        for m in metrics:
            row[f"{m}_campus"] = r[m]
            row[f"{m}_vs_district"] = r[m] - r.get(f"{m}_district_avg", np.nan)
            row[f"{m}_vs_region"] = r[m] - r.get(f"{m}_region_avg", np.nan)
        bench_rows.append(row)
bench_df = pd.DataFrame(bench_rows)

# ---------- EQUITY / SUBGROUP GAP ANALYSIS (robust) ----------
# Auto-detect a subgroup column if possible
possible_subgroups = ["Group", "group", "Ethnicity", "ethnicity", "Subgroup", "subgroup"]
SUBGROUP_COLS = [c for c in possible_subgroups if c in df.columns]
equity_diag = pd.DataFrame()

if not SUBGROUP_COLS:
    # No subgroup column found — create a single-row sheet explaining this
    equity_diag = pd.DataFrame([{
        "note": "No subgroup column found. Checked for: " + ", ".join(possible_subgroups),
        "action": "Add a subgroup column (e.g., 'Group' or 'Ethnicity') or update SUBGROUP_COLS."
    }])
    print("No subgroup column found; writing diagnostic sheet explaining the issue.")
else:
    SUBGROUP_COL = SUBGROUP_COLS[0]  # use first detected subgroup column
    EQUITY_METRICS = metrics.copy()

    # Keep district/region present (already ensured earlier)
    grp_cols = [COL_CAMPUS, COL_DIST, COL_REGION, COL_YEAR, SUBGROUP_COL]

    # 1) subgroup mean per campus-year-subgroup including district & region
    subgroup_mean = df.groupby(grp_cols)[EQUITY_METRICS].mean().reset_index().rename(
        columns={m: f"{m}_subgrp" for m in EQUITY_METRICS}
    )

    # 2) campus mean (including district & region)
    campus_mean = df.groupby([COL_CAMPUS, COL_DIST, COL_REGION, COL_YEAR])[EQUITY_METRICS].mean().reset_index().rename(
        columns={m: f"{m}_campus" for m in EQUITY_METRICS}
    )

    # 3) district and region means (yearly)
    district_mean = df.groupby([COL_DIST, COL_YEAR])[EQUITY_METRICS].mean().reset_index().rename(
        columns={m: f"{m}_district" for m in EQUITY_METRICS}
    )
    region_mean = df.groupby([COL_REGION, COL_YEAR])[EQUITY_METRICS].mean().reset_index().rename(
        columns={m: f"{m}_region" for m in EQUITY_METRICS}
    )

    # Merge: subgroup -> campus -> district -> region (keys include district and region)
    s = subgroup_mean.merge(campus_mean, on=[COL_CAMPUS, COL_DIST, COL_REGION, COL_YEAR], how="left")
    s = s.merge(district_mean, on=[COL_DIST, COL_YEAR], how="left")
    s = s.merge(region_mean, on=[COL_REGION, COL_YEAR], how="left")

    # 4) subgroup gaps
    for m in EQUITY_METRICS:
        s[f"{m}_gap_vs_campus"] = s[f"{m}_subgrp"] - s.get(f"{m}_campus")
        s[f"{m}_gap_vs_district"] = s[f"{m}_subgrp"] - s.get(f"{m}_district")
        s[f"{m}_gap_vs_region"] = s[f"{m}_subgrp"] - s.get(f"{m}_region")

    # 5) 3-year rolling subgroup means and gap trends
    s = s.sort_values([COL_CAMPUS, COL_YEAR, SUBGROUP_COL]).reset_index(drop=True)
    roll_group_keys = [COL_CAMPUS, SUBGROUP_COL]
    for m in EQUITY_METRICS:
        rm_col = f"{m}_subgrp_3yr"
        s[rm_col] = s.groupby(roll_group_keys)[f"{m}_subgrp"].rolling(window=3, min_periods=1).mean().reset_index(level=list(range(len(roll_group_keys))), drop=True)
        s[f"{m}_gap_trend_vs_district"] = (s[rm_col] - s.get(f"{m}_district")) - (s[f"{m}_subgrp"] - s.get(f"{m}_district"))

    # 6) summarize largest gaps and trends
    gap_cols = [c for c in s.columns if c.endswith("_gap_vs_district")]
    trend_cols = [c for c in s.columns if c.endswith("_gap_trend_vs_district")]
    if gap_cols:
        s["equity_gap_raw"] = s[gap_cols].abs().max(axis=1).fillna(0.0)
        s["equity_gap_trend_raw"] = s[trend_cols].abs().max(axis=1).fillna(0.0)

        def _norm(x):
            x = pd.to_numeric(x, errors="coerce").fillna(0.0)
            mn, mx = x.min(), x.max()
            if pd.isna(mn) or pd.isna(mx) or mx == mn:
                return pd.Series(0.0, index=x.index)
            return (x - mn) / (mx - mn)

        s["disparity_raw"] = s["equity_gap_raw"]
        s["disparity_c"] = _norm(s["equity_gap_raw"])
        s["disparity_trend_raw"] = s["equity_gap_trend_raw"]
        s["disparity_trend_c"] = _norm(s["equity_gap_trend_raw"])
    else:
        s["disparity_raw"] = 0.0
        s["disparity_c"] = 0.0
        s["disparity_trend_raw"] = 0.0
        s["disparity_trend_c"] = 0.0

    # prepare equity_diag for output
    keep_cols = [COL_CAMPUS, COL_YEAR, SUBGROUP_COL, "disparity_raw", "disparity_c", "disparity_trend_raw", "disparity_trend_c"]
    extra = gap_cols + trend_cols
    for c in extra:
        if c not in s.columns:
            continue
        keep_cols.append(c)
    equity_diag = s[keep_cols].copy()

# ---------- SAVE OUTPUT ----------
print("Saving results to:", OUTPUT_PATH)
with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="full_dataset_with_metrics", index=False)
    risk_cols = [COL_CAMPUS, COL_YEAR, "risk_score",
                 "erw_risk_c", "math_risk_c", "act_eng_risk_c", "act_math_risk_c",
                 "tsi_risk_c", "part_risk_c", "decline_c",
                 "sat_risk_score", "act_risk_score", "tsi"]
    risk_cols_present = [c for c in risk_cols if c in df.columns]
    df[risk_cols_present].to_excel(writer, sheet_name="risk_index", index=False)
    if not bench_df.empty:
        bench_df.to_excel(writer, sheet_name="benchmarks", index=False)
    # write equity diagnostics (always write something)
    equity_diag.to_excel(writer, sheet_name="equity_diagnostic", index=False)

print("Saved workbook with sheets: full_dataset_with_metrics, risk_index" + (", benchmarks" if not bench_df.empty else "") + ", equity_diagnostic")
print()

# ---------- QUICK SUMMARY ----------
print("\nTop 10 highest-risk campus-year (sample):")
display_cols = [COL_CAMPUS, COL_YEAR, "risk_score", COL_ERW, COL_MATH, COL_ACT_ENG, COL_ACT_MATH, COL_TSI, COL_PART, "sat_risk_score", "act_risk_score", "tsi"]
present_display_cols = [c for c in display_cols if c in df.columns]
print(df.sort_values("risk_score", ascending=False)[present_display_cols].head(10).to_string(index=False))
print("\nPhase 2 with robust equity diagnostics complete.")
