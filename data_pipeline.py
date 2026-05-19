"""
data_pipeline.py
================
MODULE 1: Real-World Data Preprocessing Pipeline
=================================================
AI Multi-Agent Traffic Simulation System — University Lab Final Project

Dataset: UCI Metro Interstate Traffic Volume (REAL DATA)
  Source : https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
  File   : Metro_Interstate_Traffic_Volume.csv.gz  (inside the .zip you downloaded)
  Records: 48,204 hourly observations (2012-2018), I-94 ATR 301 sensor, Minneapolis MN

Core AI Concepts Demonstrated:
  - Data Wrangling & Feature Engineering
  - Categorical Encoding (Label Encoding + One-Hot Encoding)
  - Feature Scaling via StandardScaler (zero-mean, unit-variance normalization)
  - Missing Value & Outlier Handling (temperature zero-clamp, IQR clipping)
  - Target variable discretization into congestion classes

Raw Columns:
  holiday            -- string: holiday name or NaN (no holiday)
  temp               -- float: temperature in Kelvin  (0.0 = bad sensor reading)
  rain_1h            -- float: rain in mm last hour
  snow_1h            -- float: snow in mm last hour
  clouds_all         -- int  : cloud coverage 0-100 %
  weather_main       -- str  : primary weather category
  weather_description-- str  : detailed weather (dropped -- redundant)
  date_time          -- str  : "YYYY-MM-DD HH:MM:SS"
  traffic_volume     -- int  : vehicles per hour -> becomes Congestion_Level target

Output: traffic_clean.csv
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RAW_CSV_OPTIONS = [
    "Metro_Interstate_Traffic_Volume.csv",
    "Metro_Interstate_Traffic_Volume.csv.gz",
]
OUTPUT_CLEAN = "traffic_clean.csv"

# Congestion thresholds (vehicles / hour)
# Derived from real dataset quartiles: Q1=1193, median=3380, Q3=4933, max=7280
FREE_FLOW_MAX = 1500   # 0 -- Free Flow
MODERATE_MAX  = 4000   # 1 -- Moderate  |  >4000 => 2 -- Heavy


# ---------------------------------------------------------------------------
# 1. Loader
# ---------------------------------------------------------------------------
def load_raw_data():
    """
    Load the UCI Metro Interstate Traffic Volume CSV.
    Accepts both plain .csv and .csv.gz formats.
    Place the file in the same folder as this script.
    """
    for fname in RAW_CSV_OPTIONS:
        if os.path.exists(fname):
            log.info("Loading real-world dataset from '%s'...", fname)
            df = pd.read_csv(fname)
            log.info("Loaded. Shape: %s", df.shape)
            return df

    raise FileNotFoundError(
        "Could not find the UCI dataset. Please place one of the following "
        "files in the same folder as this script:\n  " + str(RAW_CSV_OPTIONS) + "\n"
        "Download from: https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
    )


# ---------------------------------------------------------------------------
# 2. Target Engineering
# ---------------------------------------------------------------------------
def assign_congestion_level(traffic_volume):
    """
    Discretize continuous traffic_volume into three congestion classes.

        0 -- Free Flow        : volume <= 1,500 veh/hr
        1 -- Moderate         : 1,500 < volume <= 4,000
        2 -- Heavy Congestion : volume > 4,000
    """
    conditions = [
        traffic_volume <= FREE_FLOW_MAX,
        (traffic_volume > FREE_FLOW_MAX) & (traffic_volume <= MODERATE_MAX),
        traffic_volume > MODERATE_MAX,
    ]
    return pd.Series(
        np.select(conditions, [0, 1, 2], default=1),
        index=traffic_volume.index,
        name="Congestion_Level",
        dtype=int,
    )


# ---------------------------------------------------------------------------
# 3. Preprocessing Pipeline
# ---------------------------------------------------------------------------
def preprocess(df):
    """
    Full preprocessing pipeline applied to the real UCI dataset.

    Steps:
      1.  Drop duplicate rows.
      2.  Drop 'weather_description' (redundant with weather_main).
      3.  Parse date_time -> Hour, Day_of_Week, Month.
      4.  Fix holiday NaN -> "None" string.
      5.  Fix temp=0.0 K sensor errors -> median imputation.
      6.  Median imputation for any remaining numeric NaNs.
      7.  Clip rain_1h / snow_1h at 99th percentile.
      8.  LabelEncode 'holiday'.
      9.  One-hot encode 'weather_main'.
      10. Build Congestion_Level target; drop raw traffic_volume.
      11. StandardScaler on all numeric feature columns.
    """
    log.info("=" * 55)
    log.info("Starting preprocessing pipeline on real UCI dataset")
    log.info("=" * 55)

    # Step 1 -- Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    log.info("[1] Dropped %d duplicate rows. Remaining: %d", before - len(df), len(df))

    # Step 2 -- Drop redundant column
    if "weather_description" in df.columns:
        df = df.drop(columns=["weather_description"])
        log.info("[2] Dropped 'weather_description'.")

    # Step 3 -- Temporal features
    df["date_time"]   = pd.to_datetime(df["date_time"])
    df["Hour"]        = df["date_time"].dt.hour
    df["Day_of_Week"] = df["date_time"].dt.dayofweek
    df["Month"]       = df["date_time"].dt.month
    df = df.drop(columns=["date_time"])
    log.info("[3] Extracted Hour, Day_of_Week, Month from date_time.")

    # Step 4 -- Holiday: NaN means no holiday
    n_holiday_nan = df["holiday"].isna().sum()
    df["holiday"]  = df["holiday"].fillna("None")
    log.info("[4] Filled %d holiday NaNs with 'None'.", n_holiday_nan)

    # Step 5 -- Temperature: 0.0 K is physically impossible (sensor error)
    n_zero_temp = (df["temp"] == 0.0).sum()
    if n_zero_temp > 0:
        temp_median = df.loc[df["temp"] > 0, "temp"].median()
        df.loc[df["temp"] == 0.0, "temp"] = temp_median
        log.info("[5] Replaced %d zero-temp errors with median=%.2f K.", n_zero_temp, temp_median)
    else:
        log.info("[5] No zero-temp sensor errors found.")

    # Step 6 -- Median imputation for remaining numeric NaNs
    numeric_cols = ["temp", "rain_1h", "snow_1h", "clouds_all"]
    for col in numeric_cols:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            median_val = df[col].median()
            df[col]    = df[col].fillna(median_val)
            log.info("[6] Imputed %d NaNs in '%s' with median=%.4f.", n_null, col, median_val)
    log.info("[6] Median imputation complete.")

    # Step 7 -- Outlier clipping (99th pct) on precipitation
    for col in ["rain_1h", "snow_1h"]:
        hi      = df[col].quantile(0.99)
        clipped = (df[col] > hi).sum()
        df[col] = df[col].clip(upper=hi)
        log.info("[7] Clipped %d outliers in '%s' at 99th pct=%.4f.", clipped, col, hi)

    # Step 8 -- LabelEncode holiday
    le = LabelEncoder()
    df["Holiday_Encoded"] = le.fit_transform(df["holiday"])
    df = df.drop(columns=["holiday"])
    log.info("[8] LabelEncoded 'holiday'. Classes (%d): %s", len(le.classes_), list(le.classes_))

    # Step 9 -- One-hot encode weather_main
    df = pd.get_dummies(df, columns=["weather_main"], prefix="Weather", dtype=int)
    weather_cols = [c for c in df.columns if c.startswith("Weather_")]
    log.info("[9] One-hot encoded 'weather_main'. Columns: %s", weather_cols)

    # Step 10 -- Target variable
    df["Congestion_Level"] = assign_congestion_level(df["traffic_volume"])
    df = df.drop(columns=["traffic_volume"])
    class_dist = df["Congestion_Level"].value_counts().sort_index()
    pct = (class_dist / len(df) * 100).round(1)
    log.info(
        "[10] Congestion_Level:\n"
        "       0 Free Flow  : %d (%.1f%%)\n"
        "       1 Moderate   : %d (%.1f%%)\n"
        "       2 Heavy      : %d (%.1f%%)",
        class_dist.get(0, 0), pct.get(0, 0.0),
        class_dist.get(1, 0), pct.get(1, 0.0),
        class_dist.get(2, 0), pct.get(2, 0.0),
    )

    # Step 11 -- StandardScaler
    scale_cols = [
        c for c in df.columns
        if c != "Congestion_Level" and pd.api.types.is_numeric_dtype(df[c])
    ]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    log.info("[11] StandardScaler applied to %d feature columns.", len(scale_cols))

    log.info("Preprocessing complete. Final shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 4. Entrypoint
# ---------------------------------------------------------------------------
def run_pipeline():
    """Load real UCI data -> preprocess -> save traffic_clean.csv."""
    raw_df   = load_raw_data()
    clean_df = preprocess(raw_df.copy())
    clean_df.to_csv(OUTPUT_CLEAN, index=False)
    log.info("Clean dataset saved -> '%s'  (%d rows, %d cols)",
             OUTPUT_CLEAN, len(clean_df), clean_df.shape[1])
    log.info("\n=== Feature Summary ===\n%s", clean_df.describe().round(3).to_string())
    return clean_df


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("MODULE 1 -- Real-World Data Preprocessing Pipeline")
    log.info("=" * 60)
    processed = run_pipeline()
    log.info("Pipeline finished. '%s' is ready for Module 2.", OUTPUT_CLEAN)
