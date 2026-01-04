"""Shared utilities and configuration for the ML pipeline."""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_PATH = BASE_DIR / "dataset_from_students_only_mode_v1.csv"
CLEAN_DATA_PATH = BASE_DIR / "EDA_Output" / "dataset_cleaned.csv"
RANDOM_STATE = 42
MIN_SAMPLES = 15


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
