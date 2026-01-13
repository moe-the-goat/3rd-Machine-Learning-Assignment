"""Task 2: Exploratory Data Analysis - Dataset statistics and visualizations."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

from utils import BASE_DIR, DATA_PATH, MIN_SAMPLES, CLEAN_DATA_PATH

OUTPUT_DIR = BASE_DIR / "Results" / "Task2_EDA"
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_preprocess(data_path):
    """Load dataset and add text statistics."""
    df = pd.read_csv(data_path, encoding="utf-8-sig", dtype="string")
    
    df["Description"] = df["Description"].astype("string").fillna("").str.strip()
    df["Description"] = df["Description"].str.replace(r"\s+", " ", regex=True)
    df["Country"] = df["Country"].astype("string").fillna("").str.strip()
    df["Country_norm"] = df["Country"].str.lower()
    df["Country_display"] = df["Country_norm"].str.title()
    
    df["word_count"] = df["Description"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["Description"].apply(lambda x: len(str(x)))
    df["avg_word_length"] = df.apply(
        lambda row: row["char_count"] / max(row["word_count"], 1), axis=1
    )
    return df


def filter_dataset(df, min_samples=MIN_SAMPLES):
    """Filter to countries with sufficient samples."""
    country_counts = df["Country_display"].replace("", pd.NA).dropna().value_counts()
    valid_countries = country_counts[country_counts >= min_samples].index
    df_filtered = df[df["Country_display"].isin(valid_countries)].copy()
    return df_filtered, country_counts

def generate_summary_statistics(df, df_filtered, country_counts, output_dir):
    """Generate dataset summary report."""
    summary_lines = [
        "=" * 80,
        "DATASET SUMMARY STATISTICS",
        "=" * 80,
        "",
        "FULL DATASET:",
        f"  Total rows: {len(df)}",
        f"  Total columns: {len(df.columns)}",
        f"  Unique countries: {df['Country_display'].replace('', pd.NA).nunique(dropna=True)}",
    ]
    
    missing_country = (df["Country_display"].isna() | (df["Country_display"].str.len() == 0)).sum()
    missing_desc = (df["Description"].isna() | (df["Description"].str.len() == 0)).sum()
    summary_lines.extend([
        f"  Missing Country: {int(missing_country)}",
        f"  Missing Description: {int(missing_desc)}",
        "",
        f"FILTERED DATASET (≥{MIN_SAMPLES} samples per country):",
        f"  Total rows: {len(df_filtered)}",
        f"  Number of classes: {df_filtered['Country_display'].nunique()}",
        "",
        "DESCRIPTION LENGTH STATISTICS (filtered):",
    ])
    
    word_counts = df_filtered["word_count"]
    summary_lines.extend([
        f"  Mean words: {word_counts.mean():.2f}",
        f"  Median words: {word_counts.median():.2f}",
        f"  Std words: {word_counts.std():.2f}",
        f"  Min words: {word_counts.min()}",
        f"  Max words: {word_counts.max()}",
        "",
        "CLASS DISTRIBUTION (filtered):",
    ])
    
    for country, count in df_filtered["Country_display"].value_counts().items():
        pct = count / len(df_filtered) * 100
        summary_lines.append(f"  {country}: {count} ({pct:.1f}%)")
    
    class_counts = df_filtered["Country_display"].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    summary_lines.extend([
        "",
        "CLASS IMBALANCE ANALYSIS:",
        f"  Largest class: {class_counts.idxmax()} ({class_counts.max()} samples)",
        f"  Smallest class: {class_counts.idxmin()} ({class_counts.min()} samples)",
        f"  Imbalance ratio: {imbalance_ratio:.2f}:1",
    ])
    
    summary_text = "\n".join(summary_lines)
    with open(output_dir / "dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(summary_text)
    return summary_text

def create_visualizations(df_filtered, country_counts, output_dir):
    """Generate and save all EDA visualizations."""
    
    # Class distribution bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    class_dist = df_filtered["Country_display"].value_counts().sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(class_dist)))
    class_dist.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Country", fontsize=12)
    ax.set_title("Class Distribution: Samples per Country", fontsize=14, fontweight='bold')
    for i, v in enumerate(class_dist.values):
        ax.text(v + 0.5, i, str(v), va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()
    
    # Long-tail distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_counts = df_filtered["Country_display"].value_counts().values
    ax.plot(range(len(sorted_counts)), sorted_counts, 'o-', linewidth=2, markersize=8)
    ax.fill_between(range(len(sorted_counts)), sorted_counts, alpha=0.3)
    ax.set_xlabel("Country Rank (sorted by frequency)", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Class Frequency Distribution (Long-tail Analysis)", fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(sorted_counts)))
    ax.set_xticklabels(df_filtered["Country_display"].value_counts().index, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "class_imbalance_plot.png", dpi=150)
    plt.close()
    
    # Word count distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df_filtered["word_count"], bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(df_filtered["word_count"].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df_filtered["word_count"].mean():.1f}')
    axes[0].axvline(df_filtered["word_count"].median(), color='green', linestyle='--', 
                    label=f'Median: {df_filtered["word_count"].median():.1f}')
    axes[0].set_xlabel("Word Count", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Distribution of Description Lengths", fontsize=14, fontweight='bold')
    axes[0].legend()
    
    df_sorted = df_filtered.copy()
    country_order = df_filtered["Country_display"].value_counts().index
    df_sorted["Country_display"] = pd.Categorical(df_sorted["Country_display"], 
                                                   categories=country_order, ordered=True)
    df_sorted.boxplot(column="word_count", by="Country_display", ax=axes[1])
    axes[1].set_xlabel("Country", fontsize=12)
    axes[1].set_ylabel("Word Count", fontsize=12)
    axes[1].set_title("Word Count by Country", fontsize=14, fontweight='bold')
    plt.suptitle("")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "description_length_analysis.png", dpi=150)
    plt.close()
    
    # Average word count by country
    fig, ax = plt.subplots(figsize=(12, 6))
    stats = df_filtered.groupby("Country_display")["word_count"].agg(['mean', 'std']).sort_values('mean', ascending=True)
    stats['mean'].plot(kind='barh', ax=ax, xerr=stats['std'], capsize=3, color='steelblue', alpha=0.7)
    ax.set_xlabel("Average Word Count (± std)", fontsize=12)
    ax.set_ylabel("Country", fontsize=12)
    ax.set_title("Average Description Length by Country", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "word_count_by_country.png", dpi=150)
    plt.close()
    
    # Text features correlation
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_cols = df_filtered[["word_count", "char_count", "avg_word_length"]].copy()
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Correlation Matrix: Text Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "text_features_correlation.png", dpi=150)
    plt.close()
    
    # Categorical distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    categorical_cols = ["Weather", "Time of Day", "Season", "Mood/Emotion"]
    for idx, col in enumerate(categorical_cols):
        row, col_idx = idx // 2, idx % 2
        if col in df_filtered.columns:
            counts = df_filtered[col].value_counts()
            counts.plot(kind='bar', ax=axes[row, col_idx], color='steelblue', alpha=0.7)
            axes[row, col_idx].set_title(f"Distribution: {col}", fontsize=12, fontweight='bold')
            axes[row, col_idx].set_xlabel("")
            axes[row, col_idx].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "categorical_distributions.png", dpi=150)
    plt.close()
    
    print(f"✅ Visualizations saved to: {output_dir}")

def get_common_words_by_country(df_filtered, top_n=10):
    """Analyze most common words per country."""
    common_words = {}
    
    for country in df_filtered["Country_display"].unique():
        texts = df_filtered[df_filtered["Country_display"] == country]["Description"].str.lower()
        all_words = []
        for text in texts:
            # Simple tokenization, remove punctuation
            words = re.findall(r'\b[a-z]{3,}\b', str(text))
            all_words.extend(words)
        
        # Remove common stop words
        stop_words = {'the', 'and', 'with', 'for', 'that', 'this', 'from', 'are', 'was', 
                      'were', 'been', 'being', 'have', 'has', 'had', 'having', 'its',
                      'image', 'clear', 'shows', 'showing'}
        words_filtered = [w for w in all_words if w not in stop_words]
        common = Counter(words_filtered).most_common(top_n)
        common_words[country] = common
    
    return common_words

def save_eda_data(df_filtered, country_counts, output_dir):
    """Save processed data files."""
    # Save filtered country counts
    df_filtered["Country_display"].value_counts().to_csv(
        output_dir / "country_counts_filtered.csv",
        header=["count"],
        encoding="utf-8-sig"
    )
    
    # Save all country counts
    country_counts.to_csv(
        output_dir / "country_counts_all.csv",
        header=["count"],
        encoding="utf-8-sig"
    )
    
    # Save filtered dataset
    df_filtered.to_csv(
        output_dir / "dataset_filtered.csv",
        index=False,
        encoding="utf-8-sig"
    )
    
    # Also save cleaned dataset for other tasks to use
    CLEAN_DATA_PATH.parent.mkdir(exist_ok=True)
    df_filtered.to_csv(CLEAN_DATA_PATH, index=False, encoding="utf-8-sig")


def run_eda():
    """Main EDA pipeline."""
    print("=" * 80)
    print("TASK 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n[1/5] Loading and preprocessing data...")
    df = load_and_preprocess(DATA_PATH)
    
    print("[2/5] Filtering dataset...")
    df_filtered, country_counts = filter_dataset(df, min_samples=MIN_SAMPLES)
    
    print("[3/5] Generating summary statistics...")
    generate_summary_statistics(df, df_filtered, country_counts, OUTPUT_DIR)
    
    print("[4/5] Creating visualizations...")
    create_visualizations(df_filtered, country_counts, OUTPUT_DIR)
    
    print("[5/5] Saving data files...")
    save_eda_data(df_filtered, country_counts, OUTPUT_DIR)
    
    # Analyze common words
    print("\nMost common words per country (excluding stop words):")
    common_words = get_common_words_by_country(df_filtered)
    for country, words in sorted(common_words.items()):
        word_str = ", ".join([f"{w}({c})" for w, c in words[:5]])
        print(f"  {country}: {word_str}")
    
    print("\n" + "=" * 80)
    print("EDA COMPLETE!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    return df_filtered

if __name__ == "__main__":
    run_eda()
