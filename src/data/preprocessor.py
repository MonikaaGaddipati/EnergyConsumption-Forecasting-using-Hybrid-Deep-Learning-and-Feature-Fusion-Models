
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import joblib
import datetime


def _save_dataframe_csv(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f" Saved: {p}")


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _ensure_datetime_index(df: pd.DataFrame, date_col: str = "date", time_col: str = "time", timestamp_name: str = "timestamp") -> pd.DataFrame:
    df = df.copy()

    if date_col in df.columns and time_col in df.columns:
        combined = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), dayfirst=True, errors='coerce')
        df = df.drop(columns=[date_col, time_col])
        df[timestamp_name] = combined
        df = df.set_index(timestamp_name)
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        candidates = [c for c in df.columns if "timestamp" in c or "date" in c or "time" in c]
        if candidates:
            df[candidates[0]] = pd.to_datetime(df[candidates[0]], errors='coerce', dayfirst=True)
            df = df.set_index(candidates[0])
        else:
            raise ValueError("No date/time columns found and index is not datetime-like.")

    df = df[~df.index.isna()].sort_index()
    return df


def _coerce_numeric(df: pd.DataFrame, skip_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    if skip_cols is None:
        skip_cols = []
    for col in df.columns:
        if col in skip_cols:
            continue
        try:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
        except Exception:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _auto_detect_power_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    if "global_active_power" in cols:
        return "global_active_power"
    for c in cols:
        if "global" in c and "active" in c:
            return c
    for c in cols:
        if "active" in c or "power" in c:
            return c
    raise KeyError("Could not find a suitable power column .")


def resample_series(df: pd.DataFrame, freq: Optional[str] = "h", agg: str = "mean") -> pd.DataFrame:

    if freq is None:
        return df
    if agg == "mean":
        return df.resample(freq).mean()
    elif agg == "sum":
        return df.resample(freq).sum()
    else:
        return df.resample(freq).agg(agg)


def create_energy_column(df: pd.DataFrame, source_col: Optional[str] = None, out_col: str = "energy_consumption") -> pd.DataFrame:

    df = df.copy()
    if source_col is None:
        source_col = _auto_detect_power_column(df)
        print(f"🔎 Auto-detected power column: '{source_col}'")
    if source_col not in df.columns:
        raise KeyError(f"Source column '{source_col}' not found in df columns: {list(df.columns)}")
    df[out_col] = df[source_col].astype(float)
    return df


def fill_missing_and_smooth(df: pd.DataFrame, interpolate_limit: int = 24) -> pd.DataFrame:

    df = df.copy()
    try:
        df = df.interpolate(method="time", limit=interpolate_limit)
    except Exception:

        df = df.interpolate(method="linear", limit=interpolate_limit)
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lags_and_rolls(df: pd.DataFrame, target_col: str = "energy_consumption", lags: List[int] = [24, 48, 168]) -> pd.DataFrame:

    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    df[f"{target_col}_roll_mean_24"] = df[target_col].rolling(window=24, min_periods=1).mean().shift(1)
    df[f"{target_col}_roll_std_24"] = df[target_col].rolling(window=24, min_periods=1).std().shift(1)
    return df


def ensure_holiday_flag(df: pd.DataFrame, holiday_col: str = "holiday_flag") -> pd.DataFrame:

    df = df.copy()
    if holiday_col in df.columns:
        df["is_holiday"] = df[holiday_col].astype(int)
    else:
        df["is_holiday"] = 0
    return df


def split_and_scale(df: pd.DataFrame, processed_dir: str, numeric_cols: List[str], test_size: float = 0.2, scaler_type: str = "robust") -> Tuple[pd.DataFrame, pd.DataFrame, object]:

    df_clean = df.dropna().copy()
    n = len(df_clean)
    if n == 0:
        raise ValueError("No rows left after dropna() — check preprocessing.")
    n_test = int(np.ceil(n * test_size))
    train = df_clean.iloc[:-n_test].copy()
    test = df_clean.iloc[-n_test:].copy()

    scaler = RobustScaler() if scaler_type == "robust" else MinMaxScaler()
    scaler.fit(train[numeric_cols])
    train[numeric_cols] = scaler.transform(train[numeric_cols])
    test[numeric_cols] = scaler.transform(test[numeric_cols])

    # save processed files
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    _save_dataframe_csv(train.reset_index(), Path(processed_dir) / "trains.csv")
    _save_dataframe_csv(test.reset_index(), Path(processed_dir) / "tests.csv")
    joblib.dump(scaler, Path(processed_dir) / "scaler.joblib")
    print(f" Saved scaler to {Path(processed_dir) / 'scaler.joblib'}")

    return train, test, scaler




def run_full_preprocessing(raw_df: pd.DataFrame,
                           processed_dir: str = "data/processed",
                           resample_freq: Optional[str] = "h",
                           resample_agg: str = "mean",
                           lags: Optional[List[int]] = None,
                           test_size: float = 0.2,
                           scaler_type: str = "robust",
                           interpolate_limit: int = 24,
                           fuse_weather_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    if lags is None:
        lags = [24, 48, 168]

    df = raw_df.copy()
    df = _normalize_column_names(df)
    df = _ensure_datetime_index(df, date_col="date", time_col="time")
    df = _coerce_numeric(df, skip_cols=[])
    df = resample_series(df, freq=resample_freq, agg=resample_agg)
    df = create_energy_column(df, source_col=None, out_col="energy_consumption")

    try:
        raw_hourly_path = Path("data/raw/household_power_consumption_hourly.csv")
        _save_dataframe_csv(df.reset_index(), str(raw_hourly_path))
    except Exception as e:
        print("Warning: failed to save hourly raw snapshot:", e)

    if fuse_weather_df is not None:
        wf = fuse_weather_df.copy()
        if not isinstance(wf.index, pd.DatetimeIndex):
            raise ValueError("fuse_weather_df must have a DatetimeIndex.")
        df = df.join(wf, how="left")

    df = fill_missing_and_smooth(df, interpolate_limit=interpolate_limit)

    if "energy_consumption" in df.columns:
        q1 = df["energy_consumption"].quantile(0.25)
        q3 = df["energy_consumption"].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        mask = (df["energy_consumption"] < low) | (df["energy_consumption"] > high)
        if mask.any():
            df.loc[mask, "energy_consumption"] = df["energy_consumption"].rolling(window=24, center=True, min_periods=1).median()[mask]

    df = add_time_features(df)
    df = ensure_holiday_flag(df, holiday_col="holiday_flag")
    df = add_lags_and_rolls(df, target_col="energy_consumption", lags=lags)

    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    binary_cols = ["is_weekend", "is_holiday"]
    numeric_cols = [c for c in numeric_cols if c not in binary_cols]

    train_df, test_df, scaler = split_and_scale(df, processed_dir, numeric_cols, test_size=test_size, scaler_type=scaler_type)

    try:
        report_path = Path("results/preprocessing_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write("Preprocessing report\n")
            f.write("====================\n")
            f.write(f"Generated: {datetime.datetime.now().isoformat()}\n\n")
            f.write(f"Raw input shape: {raw_df.shape}\n")
            f.write(f"After resample shape: {df.shape}\n\n")
            f.write("Columns:\n")
            f.write(", ".join(df.columns.tolist()) + "\n\n")
            f.write("Missing values (after cleaning):\n")
            f.write(str(df.isna().sum()) + "\n\n")
            f.write("Train shape: " + str(train_df.shape) + "\n")
            f.write("Test shape: " + str(test_df.shape) + "\n")
        print(f" Preprocessing report written to {report_path}")
    except Exception as e:
        print("Could not write preprocessing report:", e)

    print(" Preprocessing complete!")
    print("   - Raw hourly (saved): data/raw/household_power_consumption_hourly.csv")
    print(f"   - Train: {Path(processed_dir) / 'train.csv'}")
    print(f"   - Test : {Path(processed_dir) / 'test.csv'}")
    print(f"   - Scaler: {Path(processed_dir) / 'scaler.joblib'}")
    print(f"   - Report: results/preprocessing_report.txt")

    return train_df, test_df, scaler