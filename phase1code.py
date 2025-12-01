# Phase 1 — Data foundation (combined script)
import pandas as pd
import numpy as np
import re
from pathlib import Path
# Visualization libraries
import matplotlib.pyplot as plt 
import seaborn as sns 

# ---------- CONFIG ----------
file_path = "/Users/nehal/Desktop/test/combined_year_dataset_cleaned.xlsx"
output_path = "/Users/nehal/Desktop/test/dataset_phase1_ready.xlsx"
image_output_dir = "/Users/nehal/Desktop/test/phase1_visualizations" # Directory for saving plots

# HARDCODED COLUMN MAPPING 
found = {
    "campus": "Campus",        
    "district": "District",
    "region": "Region",
    "year": "year",
    "sat_erw": "ERW_SAT",
    "sat_math": "Math_SAT",
    "tsi": "Above_TSI_Both_Rate_SAT", 
    "participation": "Part_Rate_SAT",
    "group": "Group" 
}

# Masked values to consider (TEA style)
masked_values = ["<25", "*", "-", "NA", "N/A", "na", "n/a", "PrivacySuppressed"]

# ---------- HELPERS ----------
def sanitize_numeric_column(series):
    """Strip commas, percent signs, parentheses, convert to numeric if possible."""
    def clean_val(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, float, np.number)): return x
        s = str(x).strip()
        if s in masked_values: return np.nan
        s = re.sub(r"[,\%]|[\(\)]|\*|†|‡", "", s)
        s = re.sub(r"^[^\d\.-]+", "", s)
        s = re.sub(r"[^\d\.-]+$", "", s)
        if s == "": return np.nan
        try:
            if "." in s or re.search(r"\d+\.\d+", s): return float(s)
            return int(float(s))
        except:
            try: return float(re.sub(r"[^\d\.-]", "", s))
            except: return np.nan
    return series.map(clean_val)

def safe_diff(df, group_col, value_col, out_col):
    """Helper to compute year-over-year difference."""
    if value_col in df.columns:
        if group_col and group_col in df.columns:
            df[out_col] = df.groupby(group_col)[value_col].diff()
        else:
            df[out_col] = df[value_col].diff() 
        return True
    return False

# ---------- 1) LOAD & INITIAL CHECK ----------
print("Loading:", file_path)
df = pd.read_excel(file_path, engine="openpyxl")
print("Loaded. Shape:", df.shape)

# Define column variables
year_col = found.get("year")
campus_col = found.get("campus")
district_col = found.get("district")
region_col = found.get("region")
group_col = found.get("group")
sat_erw_col = found.get("sat_erw")
sat_math_col = found.get("sat_math")
tsi_col = found.get("tsi")
part_col = found.get("participation")
sat_total_col = "Total_SAT"

# ---------- 2) CLEAN & COERCE NUMERICS ----------
print("\nCleaning and coercing numeric columns...")
df.replace(masked_values, np.nan, inplace=True)
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in obj_cols:
    if col in [found.get(k) for k in ["sat_erw", "sat_math", "tsi", "participation"]]:
        df[col] = sanitize_numeric_column(df[col])
        continue
    non_null = df[col].dropna()
    if len(non_null) > 0 and non_null.astype(str).str.contains(r"\d").mean() >= 0.30:
        df[col] = sanitize_numeric_column(df[col])

# Final coercion for key metrics
for col in [sat_erw_col, sat_math_col, tsi_col, part_col]:
    if col and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"Coerced {col} -> numeric. Nulls: {df[col].isna().sum()}")

# NORMALIZE LOCATION COLUMNS (before grouping)
for col in [campus_col, district_col, region_col, group_col]:
    if col and col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# ---------- 3) SORT and determine grouping column for time series ----------
if campus_col and campus_col in df.columns:
    used_group = campus_col
elif district_col and district_col in df.columns:
    used_group = district_col
else:
    used_group = None
sort_cols = [used_group, year_col] if used_group else [year_col]
df = df.sort_values(sort_cols).reset_index(drop=True)

# ---------- 4) CREATE TIME-SERIES FEATURES (for plotting) ----------
print("\nCreating time-series features...")
diffs_created = []
if safe_diff(df, used_group, sat_erw_col, "ERW_change"): diffs_created.append("ERW_change")
# Note: Creating rolling mean columns for use in Section 11, even though the feature columns themselves aren't the final output.
rolling_cols = []
if used_group and used_group in df.columns:
    for col, name in [(sat_erw_col, "ERW_3yr"), (sat_math_col, "Math_3yr"), (tsi_col, "TSI_3yr")]:
        if col and col in df.columns:
            df[name] = df.groupby(used_group)[col].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
            rolling_cols.append(name)
print(f"Time-series features created: {diffs_created + rolling_cols}")

# ---------- 5) PREPARE AGGREGATION DATAFRAMES (for plotting) ----------
print("\nPreparing aggregation dataframes for comparative plots...")
metrics = [sat_erw_col, tsi_col, part_col] # Focus on key metrics for comparisons
agg_dfs = {}

# A. Group-level average scores
if group_col and group_col in df.columns:
    # Average performance for each student group across all years
    agg_dfs['group_avg'] = df.groupby(group_col, dropna=True)[metrics].mean().reset_index()

# B. Campus vs District Comparison (Yearly)
if campus_col and district_col and campus_col in df.columns and district_col in df.columns:
    # Calculate District Average for each year
    district_avg = df.groupby([district_col, year_col], dropna=True)[metrics].mean().reset_index()
    district_avg.columns = [district_col, year_col] + [f'{c}_Avg' for c in metrics]
    agg_dfs['district_avg'] = district_avg

# C. District vs Region Comparison (Yearly)
if district_col and region_col and district_col in df.columns and region_col in df.columns:
    # Calculate Regional Average for each year
    region_avg = df.groupby([region_col, year_col], dropna=True)[metrics].mean().reset_index()
    region_avg.columns = [region_col, year_col] + [f'{c}_Avg' for c in metrics]
    agg_dfs['region_avg'] = region_avg

print(f"Prepared {len(agg_dfs)} aggregation tables for visualization.")

# --- 6) VISUALIZATION AND SAVING ---
Path(image_output_dir).mkdir(parents=True, exist_ok=True)
print(f"\nGenerating and saving visualizations to: {image_output_dir}")

sns.set_style("whitegrid")
saved_plots = []

# --- PLOT 1: 3-YEAR ROLLING TRENDS (Line Plots) ---
for col in rolling_cols:
    plt.figure(figsize=(10, 6))
    
    # Calculate overall trend
    avg_df = df.groupby(year_col, dropna=True)[col].mean().reset_index()
    
    # Plot individual group trends (sampled for clarity)
    if used_group and used_group in df.columns:
        sample_groups = np.random.choice(df[used_group].dropna().unique(), min(10, len(df[used_group].dropna().unique())), replace=False)
        sample_df = df[df[used_group].isin(sample_groups)]
        sns.lineplot(x=year_col, y=col, hue=used_group, data=sample_df, 
                     linestyle='--', alpha=0.3, legend=False, palette='tab10')
    
    # Plot the overall trend
    sns.lineplot(x=year_col, y=col, data=avg_df, 
                 marker='o', linestyle='-', color='red', linewidth=3, label='Overall Average Trend')
    
    plt.title(f'3-Year Rolling Trend: {col}')
    plt.xlabel(year_col)
    plt.ylabel(col)
    file_name = f'Trend_1_{col}.png'
    plt.savefig(Path(image_output_dir) / file_name)
    plt.close()
    saved_plots.append(file_name)


# --- PLOT 2: GROUP PERFORMANCE COMPARISON (Bar Plot/Distribution) ---
if 'group_avg' in agg_dfs:
    plt.figure(figsize=(12, 6))
    # Focus on ERW for simplicity
    sns.barplot(x=group_col, y=sat_erw_col, data=agg_dfs['group_avg'], palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Overall Average {sat_erw_col} Score by Student Group')
    plt.ylabel(f'Average {sat_erw_col} Score')
    plt.xlabel(group_col)
    plt.tight_layout()
    file_name = f'Comparison_2_Group_Avg_{sat_erw_col}.png'
    plt.savefig(Path(image_output_dir) / file_name)
    plt.close()
    saved_plots.append(file_name)

# --- PLOT 3: CAMPUS vs DISTRICT PERFORMANCE (Box Plot of Deviation) ---
if campus_col and 'district_avg' in agg_dfs:
    # Merge district average back to calculate deviation
    temp_df = df.merge(agg_dfs['district_avg'], on=[district_col, year_col], how='left')
    deviation_col = f'{sat_erw_col}_Dev_Campus_Dist'
    temp_df[deviation_col] = temp_df[sat_erw_col] - temp_df[f'{sat_erw_col}_Avg']

    plt.figure(figsize=(12, 6))
    # Plot the distribution of Campus scores relative to their District average
    sns.histplot(temp_df[deviation_col].dropna(), kde=True, bins=50, color='skyblue')
    plt.axvline(0, color='red', linestyle='--', label='District Average')
    plt.title(f'Distribution of {sat_erw_col} Deviation (Campus Score - District Average)')
    plt.xlabel('Deviation Score')
    plt.ylabel('Count of Campuses')
    plt.legend()
    file_name = f'Comparison_3_Campus_Dist_Deviation.png'
    plt.savefig(Path(image_output_dir) / file_name)
    plt.close()
    saved_plots.append(file_name)

# --- PLOT 4: DISTRICT vs REGION PERFORMANCE (Scatter Plot) ---
if district_col and 'region_avg' in agg_dfs:
    # Flatten regional data to plot district average vs region average for one year
    plot_metric = sat_erw_col
    plot_year = df[year_col].max() # Use the most recent year

    # Filter District Averages for the latest year
    dist_latest = agg_dfs['district_avg'][agg_dfs['district_avg'][year_col] == plot_year][[district_col, f'{plot_metric}_Avg']]
    
    # Get Region Average for the latest year
    region_latest = agg_dfs['region_avg'][agg_dfs['region_avg'][year_col] == plot_year][[region_col, f'{plot_metric}_Avg']]
    
    # Merge District Avg with Region
    plot_df = dist_latest.merge(df[[district_col, region_col]].drop_duplicates(), on=district_col, how='left')
    plot_df = plot_df.merge(region_latest, on=region_col, how='left', suffixes=('_Dist', '_Reg'))
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=f'{plot_metric}_Avg_Dist', y=f'{plot_metric}_Avg_Reg', hue=region_col, data=plot_df, alpha=0.7)
    
    # Add diagonal line (where District Avg = Region Avg)
    max_val = plot_df[[f'{plot_metric}_Avg_Dist', f'{plot_metric}_Avg_Reg']].max().max()
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', label='District = Region')

    plt.title(f'District Average vs. Region Average {plot_metric} ({plot_year})')
    plt.xlabel(f'District Average {plot_metric}')
    plt.ylabel(f'Region Average {plot_metric}')
    plt.legend(title=region_col, bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    file_name = f'Comparison_4_Dist_Region_Scatter.png'
    plt.savefig(Path(image_output_dir) / file_name)
    plt.close()
    saved_plots.append(file_name)

print(f"Successfully generated and saved {len(saved_plots)} visualization files.")

# --- 7) SAVE PHASE 1 DATASET (with time-series features only) ---
# Saving the raw data + time-series features (Rolling Mean and Diff) as requested
print("\nSaving Phase 1 dataset to:", output_path)
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
df.to_excel(output_path, index=False)
print("Saved. Final shape:", df.shape)

# --- 8) BRIEF SAMPLE OUTPUT ---
print("\nPhase 1 complete. Data ready:", output_path)
print(f"Visualizations saved to: {image_output_dir}")