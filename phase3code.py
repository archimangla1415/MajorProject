import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")

# Try to import xgboost and tensorflow; if not installed, XGBoost/LSTM parts will be skipped with info.
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ---------- PATHS ----------
INPUT = "/Users/nehal/Desktop/test/phase2_analytics_with_ACT.xlsx"
OUTPUT = "/Users/nehal/Desktop/test/phase3_predictions.xlsx"
MODEL_DIR = Path("/Users/nehal/Desktop/test/phase3_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- SETTINGS ----------
LAG_YEARS = [1, 2, 3]   # create 1-,2-,3-year lags
RANDOM_STATE = 42

# Targets to predict (ONLY SAT & ACT subjects now)
TARGET_CANDIDATES = [
    ("ERW_SAT", "sat_erw"),
    ("Math_SAT", "sat_math"),
    ("English_ACT", "act_eng"),
    ("Math_ACT", "act_math"),
    ("Total_SAT", "sat_total"),
    ("TSI_readiness", "tsi_ready"),
]

# Evaluation helpers
def adj_r2_score(y_true, y_pred, p):
    """Adjusted R2: p = number of predictors (features)"""
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    if n - p - 1 == 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# ----- NEW: accuracy helpers -----
def pct_within_abs(y_true, y_pred, abs_tol):
    """Percent of predictions within ±abs_tol units of actual."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    valid = ~np.isnan(y_true)
    if valid.sum() == 0:
        return np.nan
    return (np.abs(y_true[valid] - y_pred[valid]) <= abs_tol).sum() / valid.sum() * 100.0

def pct_within_pct(y_true, y_pred, pct_tol=5.0):
    """Percentage of predictions within ±pct_tol percent of actuals."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-8
    denom = np.where(np.abs(y_true) < eps, np.nan, np.abs(y_true))
    rel_err = np.abs(y_true - y_pred) / denom * 100.0
    valid = ~np.isnan(rel_err)
    if valid.sum() == 0:
        return np.nan
    return (rel_err[valid] <= pct_tol).sum() / valid.sum() * 100.0

# ---------- LOAD ----------
print("Loading data:", INPUT)
df = pd.read_excel(INPUT, engine="openpyxl")
print("Shape:", df.shape)

# Ensure year & campus exist
if "year" not in df.columns or "Campus" not in df.columns:
    raise ValueError("Columns 'year' and 'Campus' are required in the dataset.")

# sort and ensure proper types
df = df.sort_values(["Campus", "year"]).reset_index(drop=True)
df["year"] = df["year"].astype(int)

# ---------- LAG FEATURE CREATION ----------
# For each numeric metric (selected targets) create lagged features per campus
lag_cols = []
for col in [c for c, _ in TARGET_CANDIDATES if c in df.columns]:
    for lag in LAG_YEARS:
        newcol = f"{col}_lag{lag}"
        df[newcol] = df.groupby("Campus")[col].shift(lag)
        lag_cols.append(newcol)

# Additional features: year (as numeric)
df["year_norm"] = df["year"]  # can be scaled later

latest_year = df["year"].max()
print("Latest data year:", latest_year)

# Build base features list
base_features = ["year_norm"] + lag_cols

# Replace infinite / very large values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

predictions_list = []
metrics_list = []

# For each target that exists, build dataset and train models
for target_col, target_short in TARGET_CANDIDATES:
    if target_col not in df.columns:
        print(f"Skipping target {target_col} (not present).")
        continue
    print("\n---- Modeling target:", target_col, "----")
    features = [f for f in base_features if f in df.columns]
    if len([f for f in features if "_lag" in f]) == 0:
        print("  No lag features available — skipping target.")
        continue

    model_df = df[features + [target_col, "Campus", "year"]].copy()
    model_df = model_df[~model_df[target_col].isna()]
    model_df = model_df.dropna(subset=[c for c in features if "_lag" in c], how='all').reset_index(drop=True)
    if model_df.shape[0] < 50:
        print("  Not enough rows for reliable modeling (rows:", model_df.shape[0], ") — still proceeding but results may be unstable.")

    train_df = model_df[model_df["year"] < latest_year].copy()
    test_df  = model_df[model_df["year"] == latest_year].copy()
    if test_df.shape[0] == 0:
        train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=RANDOM_STATE)
        print("  No rows for latest year; using random 80/20 split.")

    X_train = train_df[features].fillna(0.0)
    y_train = train_df[target_col].astype(float).fillna(0.0)
    X_test  = test_df[features].fillna(0.0)
    y_test  = test_df[target_col].astype(float).fillna(0.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, MODEL_DIR / f"{target_short}_scaler.pkl")

    # All remaining targets are score targets => absolute tolerance for accuracy
    abs_tol = 20.0  # ±20 points for SAT/ACT

    # ---------- MODEL 1: LINEAR REGRESSION ----------
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
    r2_lr = r2_score(y_test, y_pred_lr)
    adjr2_lr = adj_r2_score(y_test, y_pred_lr, p=X_train.shape[1])
    accuracy_lr = pct_within_abs(y_test, y_pred_lr, abs_tol)
    print(f"  LinearRegression -> MAE: {mae_lr:.3f}, RMSE: {rmse_lr:.3f}, R2: {r2_lr:.3f}, AdjR2: {adjr2_lr:.3f}, accuracy% (±{abs_tol}): {accuracy_lr if not np.isnan(accuracy_lr) else 'NA'}")
    joblib.dump(lr, MODEL_DIR / f"{target_short}_linearreg.joblib")

    # ---------- MODEL 2: RANDOM FOREST ----------
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train.fillna(0.0), y_train)
    y_pred_rf = rf.predict(X_test.fillna(0.0))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
    r2_rf = r2_score(y_test, y_pred_rf)
    adjr2_rf = adj_r2_score(y_test, y_pred_rf, p=X_train.shape[1])
    accuracy_rf = pct_within_abs(y_test, y_pred_rf, abs_tol)
    print(f"  RandomForest -> MAE: {mae_rf:.3f}, RMSE: {rmse_rf:.3f}, R2: {r2_rf:.3f}, AdjR2: {adjr2_rf:.3f}, accuracy% (±{abs_tol}): {accuracy_rf if not np.isnan(accuracy_rf) else 'NA'}")
    joblib.dump(rf, MODEL_DIR / f"{target_short}_rf.joblib")

    # ---------- MODEL 3: XGBOOST ----------
    if HAS_XGB:
        xgbr = xgb.XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, tree_method='approx')
        xgbr.fit(X_train.fillna(0.0), y_train)
        y_pred_xgb = xgbr.predict(X_test.fillna(0.0))
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
        r2_xgb = r2_score(y_test, y_pred_xgb)
        adjr2_xgb = adj_r2_score(y_test, y_pred_xgb, p=X_train.shape[1])
        accuracy_xgb = pct_within_abs(y_test, y_pred_xgb, abs_tol)
        print(f"  XGBoost -> MAE: {mae_xgb:.3f}, RMSE: {rmse_xgb:.3f}, R2: {r2_xgb:.3f}, AdjR2: {adjr2_xgb:.3f}, accuracy% (±{abs_tol}): {accuracy_xgb if not np.isnan(accuracy_xgb) else 'NA'}")
        xgbr.save_model(str(MODEL_DIR / f"{target_short}_xgb.model"))
    else:
        print("  XGBoost not installed — skipping XGBoost for this target.")
        y_pred_xgb = None
        mae_xgb = rmse_xgb = r2_xgb = adjr2_xgb = accuracy_xgb = None

    
    # ---------- AGGREGATE PREDICTIONS FRAME ----------
    preds_df = test_df[["Campus", "year"] + features].copy().reset_index(drop=True)
    preds_df["actual"] = y_test.values
    preds_df["pred_lr"] = y_pred_lr
    preds_df["pred_rf"] = y_pred_rf
    preds_df["pred_xgb"] = y_pred_xgb if HAS_XGB else np.nan # mapping sequences back is nontrivial; leave blank
    preds_df[target_col] = target_col
    predictions_list.append((target_col, preds_df))

    # ---------- METRICS (include accuracy) ----------
    metrics_list.append({
        "target": target_col, "model": "LinearRegression",
        "MAE": mae_lr, "RMSE": rmse_lr, "R2": r2_lr, "AdjR2": adjr2_lr,
        "accuracy_pct": accuracy_lr
    })
    metrics_list.append({
        "target": target_col, "model": "RandomForest",
        "MAE": mae_rf, "RMSE": rmse_rf, "R2": r2_rf, "AdjR2": adjr2_rf,
        "accuracy_pct": accuracy_rf
    })
    if HAS_XGB:
        metrics_list.append({
            "target": target_col, "model": "XGBoost",
            "MAE": mae_xgb, "RMSE": rmse_xgb, "R2": r2_xgb, "AdjR2": adjr2_xgb,
            "accuracy_pct": accuracy_xgb
        })
  
# ---------- SAVE PREDICTIONS & METRICS ----------
with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="full_dataset", index=False)
    for target, preds in predictions_list:
        safe_name = f"preds_{target}".replace("/", "_").replace(" ", "_")
        preds.to_excel(writer, sheet_name=safe_name, index=False)
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_excel(writer, sheet_name="metrics_summary", index=False)

print("\nSaved predictions and metrics to:", OUTPUT)
print("Model artifacts folder:", MODEL_DIR)
print("Phase 3 modeling complete.")
