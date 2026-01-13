"""Shared utilities and configuration for the ML pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Paths and settings
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_PATH = BASE_DIR / "dataset_from_students_only_mode_v1.csv"
RESULTS_DIR = BASE_DIR / "Results"
CLEAN_DATA_PATH = RESULTS_DIR / "Task2_EDA" / "dataset_cleaned.csv"
RANDOM_STATE = 42
MIN_SAMPLES = 15

# Data split ratios (consistent across all tasks)
TEST_SIZE = 0.2      # 20% for testing
VAL_SIZE = 0.125     # 12.5% of remaining 80% = 10% of total


def load_raw_data():
    """Load raw dataset."""
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def load_and_prepare_data(min_samples=MIN_SAMPLES):
    """Load, clean, and filter data. Returns DataFrame with valid countries only."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["Description"] = df["Description"].str.replace(r"\s+", " ", regex=True)
    df["Country"] = df["Country"].fillna("").astype(str).str.strip()
    
    country_counts = df["Country"].value_counts()
    valid_countries = country_counts[country_counts >= min_samples].index
    df_filtered = df[df["Country"].isin(valid_countries)].copy()
    df_filtered = df_filtered[df_filtered["Description"].str.len() > 0]
    
    return df_filtered


def create_data_splits(df, include_validation=True, return_indices=False):
    """
    Create consistent stratified data splits across all tasks.
    
    Split ratios: 70% train / 10% validation / 20% test
    (When include_validation=False: 80% train / 20% test)
    
    Parameters:
        df: DataFrame with 'Description' and 'Country' columns
        include_validation: If True, returns train/val/test. If False, returns train/test only.
        return_indices: If True, also returns the original DataFrame indices for each split.
    
    Returns:
        If include_validation=True:
            (X_train, X_val, X_test, y_train, y_val, y_test) or with indices
        If include_validation=False:
            (X_train, X_test, y_train, y_test) or with indices
    """
    X = df["Description"].astype(str)
    y = df["Country"].astype(str)
    indices = df.index
    
    # First split: 80% train+val, 20% test
    if return_indices:
        X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
            X, y, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    
    if not include_validation:
        if return_indices:
            return X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test
        return X_trainval, X_test, y_trainval, y_test
    
    # Second split: 87.5% train, 12.5% val (from trainval) = 70% train, 10% val overall
    if return_indices:
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X_trainval, y_trainval, idx_trainval, 
            test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_trainval
        )
        return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, 
            test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_trainval
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


def get_tfidf_config():
    """Standard TF-IDF configuration."""
    return {
        "lowercase": True,
        "stop_words": "english",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
        "max_features": 5000
    }
