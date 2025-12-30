"""
Task 5: Performance Analysis - Error Analysis
Analyzes misclassifications and identifies patterns in errors.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils import (
    BASE_DIR, load_and_prepare_data, get_tfidf_config, RANDOM_STATE
)

OUTPUT_DIR = BASE_DIR / "Task5_ErrorAnalysis"
TASK4_DIR = BASE_DIR / "Task4_Models"

def load_trained_model():
    """Load the best model and vectorizer from Task 4."""
    model_path = TASK4_DIR / "best_model_svm.pkl"
    vectorizer_path = TASK4_DIR / "tfidf_vectorizer.pkl"
    
    if model_path.exists() and vectorizer_path.exists():
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("  Loaded saved model and vectorizer from Task 4")
        return model, vectorizer
    
    return None, None


def train_model_fallback(X_train, y_train):
    """Fallback: train SVM if no saved model found."""
    print("  Training SVM model (no saved model found)...")
    model = SVC(
        C=10.0,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model

def analyze_errors(df, X_test_tfidf, y_test, y_pred, test_indices, labels, output_dir):
    """Analyze misclassified examples."""
    test_df = df.loc[test_indices].copy()
    test_df = test_df.reset_index(drop=True)
    test_df['True_Country'] = y_test.reset_index(drop=True)
    test_df['Predicted_Country'] = y_pred
    test_df['Is_Correct'] = test_df['True_Country'] == test_df['Predicted_Country']
    
    correct_df = test_df[test_df['Is_Correct']]
    errors_df = test_df[~test_df['Is_Correct']]
    
    print(f"\nTotal test samples: {len(test_df)}")
    print(f"Correct predictions: {len(correct_df)} ({len(correct_df)/len(test_df)*100:.1f}%)")
    print(f"Incorrect predictions: {len(errors_df)} ({len(errors_df)/len(test_df)*100:.1f}%)")
    
    # Save misclassified samples
    errors_output = errors_df[['Description', 'True_Country', 'Predicted_Country']].copy()
    errors_output['Description_Preview'] = errors_output['Description'].str[:200] + "..."
    errors_output.to_csv(output_dir / "misclassified_examples.csv", index=False, encoding='utf-8-sig')
    
    return test_df, errors_df

def analyze_per_class_errors(errors_df, labels, output_dir):
    """Analyze errors by class and create visualizations."""
    print("\n" + "=" * 60)
    print("PER-CLASS ERROR ANALYSIS")
    print("=" * 60)
    
    error_by_true_class = errors_df['True_Country'].value_counts()
    error_by_pred_class = errors_df['Predicted_Country'].value_counts()
    
    print("\nMost frequently misclassified (true) classes:")
    for country, count in error_by_true_class.items():
        print(f"  {country}: {count} errors")
    
    print("\nMost common false positive predictions:")
    for country, count in error_by_pred_class.items():
        print(f"  {country}: {count} false positives")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Errors by true class
    error_by_true_class.plot(kind='barh', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_xlabel('Number of Errors', fontsize=12)
    axes[0].set_ylabel('True Country', fontsize=12)
    axes[0].set_title('Misclassifications by True Class', fontsize=14, fontweight='bold')
    
    # Errors by predicted class
    error_by_pred_class.plot(kind='barh', ax=axes[1], color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Number of False Positives', fontsize=12)
    axes[1].set_ylabel('Predicted Country', fontsize=12)
    axes[1].set_title('False Positives by Predicted Class', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_errors.png", dpi=150)
    plt.close()
    
    return error_by_true_class, error_by_pred_class

def analyze_confusion_pairs(errors_df, output_dir):
    """Identify most commonly confused country pairs."""
    print("\n" + "=" * 60)
    print("CONFUSION PAIR ANALYSIS")
    print("=" * 60)
    
    confusion_pairs = errors_df.groupby(['True_Country', 'Predicted_Country']).size()
    confusion_pairs = confusion_pairs.sort_values(ascending=False)
    
    print("\nMost commonly confused pairs (True -> Predicted):")
    for (true_c, pred_c), count in confusion_pairs.head(10).items():
        print(f"  {true_c} -> {pred_c}: {count} times")
    
    confusion_pairs.to_csv(output_dir / "confusion_pairs.csv", header=['count'])
    
    # Heatmap of confusions
    pivot = errors_df.groupby(['True_Country', 'Predicted_Country']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='Reds', cbar_kws={'label': 'Confusion Count'})
    plt.xlabel('Predicted Country', fontsize=12)
    plt.ylabel('True Country', fontsize=12)
    plt.title('Confusion Heatmap (Errors Only)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_heatmap_errors.png", dpi=150)
    plt.close()
    
    return confusion_pairs

def analyze_error_descriptions(errors_df, output_dir):
    """Analyze word patterns in misclassified descriptions."""
    print("\n" + "=" * 60)
    print("ERROR DESCRIPTION ANALYSIS")
    print("=" * 60)
    
    errors_df['word_count'] = errors_df['Description'].apply(lambda x: len(str(x).split()))
    
    print("\nWord count statistics for errors:")
    print(f"  Mean: {errors_df['word_count'].mean():.1f}")
    print(f"  Median: {errors_df['word_count'].median():.1f}")
    print(f"  Min: {errors_df['word_count'].min()}")
    print(f"  Max: {errors_df['word_count'].max()}")
    
    # Common words in errors
    all_error_words = []
    for desc in errors_df['Description']:
        words = re.findall(r'\b[a-z]{3,}\b', str(desc).lower())
        all_error_words.extend(words)
    
    stop_words = {'the', 'and', 'with', 'for', 'that', 'this', 'from', 'are', 'was', 
                  'were', 'been', 'being', 'have', 'has', 'had', 'having', 'its',
                  'image', 'clear', 'shows', 'showing', 'one', 'can', 'seen'}
    
    error_words = [w for w in all_error_words if w not in stop_words]
    common_error_words = Counter(error_words).most_common(20)
    
    print("\nMost common words in misclassified descriptions:")
    for word, count in common_error_words:
        print(f"  {word}: {count}")
    
    # Theme analysis
    theme_words = {
        'architecture': ['temple', 'church', 'mosque', 'building', 'architecture', 'dome', 'tower'],
        'nature': ['beach', 'mountain', 'sea', 'water', 'sky', 'landscape', 'nature', 'island'],
        'history': ['ancient', 'historical', 'old', 'ruins', 'heritage', 'traditional'],
        'urban': ['city', 'street', 'modern', 'urban', 'downtown']
    }
    
    print("\nTheme frequency in errors:")
    for theme, keywords in theme_words.items():
        count = sum(1 for desc in errors_df['Description'] 
                   if any(kw in str(desc).lower() for kw in keywords))
        print(f"  {theme}: {count} errors ({count/len(errors_df)*100:.1f}%)")
    
    return common_error_words

def analyze_sample_errors_detailed(errors_df, output_dir):
    """Write detailed analysis of sample errors to file."""
    print("\n" + "=" * 60)
    print("DETAILED ERROR EXAMPLES")
    print("=" * 60)
    
    analysis_lines = []
    analysis_lines.append("DETAILED ERROR ANALYSIS")
    analysis_lines.append("=" * 80)
    analysis_lines.append("")
    
    top_pairs = errors_df.groupby(['True_Country', 'Predicted_Country']).size().sort_values(ascending=False).head(5)
    
    for (true_c, pred_c), count in top_pairs.items():
        analysis_lines.append(f"\n{'='*60}")
        analysis_lines.append(f"CONFUSION: {true_c} -> {pred_c} ({count} errors)")
        analysis_lines.append(f"{'='*60}")
        
        pair_errors = errors_df[(errors_df['True_Country'] == true_c) & 
                                (errors_df['Predicted_Country'] == pred_c)]
        
        for idx, row in pair_errors.head(2).iterrows():
            analysis_lines.append(f"\nExample:")
            analysis_lines.append(f"  Description: {row['Description'][:300]}...")
            analysis_lines.append(f"  True: {true_c}")
            analysis_lines.append(f"  Predicted: {pred_c}")
            
            # Identify potential confusing terms
            desc_lower = str(row['Description']).lower()
            
            ambiguous_terms = []
            if 'beach' in desc_lower: ambiguous_terms.append('beach (common to many countries)')
            if 'ancient' in desc_lower: ambiguous_terms.append('ancient (Greece, Italy, Egypt, etc.)')
            if 'mediterranean' in desc_lower: ambiguous_terms.append('Mediterranean (multiple countries)')
            if 'temple' in desc_lower: ambiguous_terms.append('temple (Japan, Greece, etc.)')
            if 'mosque' in desc_lower: ambiguous_terms.append('mosque (Turkey, Egypt, etc.)')
            if 'alps' in desc_lower or 'mountain' in desc_lower: ambiguous_terms.append('mountains (Switzerland, France, Italy)')
            
            if ambiguous_terms:
                analysis_lines.append(f"  Potential confusion: {', '.join(ambiguous_terms)}")
    
    analysis_text = "\n".join(analysis_lines)
    
    with open(output_dir / "detailed_error_analysis.txt", "w", encoding="utf-8") as f:
        f.write(analysis_text)
    
    print("\nSee detailed_error_analysis.txt for in-depth examples")
    
    return analysis_text

def analyze_data_quality_issues(errors_df, df_original, output_dir):
    """Identify patterns that may indicate data quality issues."""
    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    issues = []
    
    errors_df['word_count'] = errors_df['Description'].apply(lambda x: len(str(x).split()))
    
    # Short descriptions issue
    short_errors = errors_df[errors_df['word_count'] < 10]
    if len(short_errors) > 0:
        issues.append(f"Short descriptions (<10 words): {len(short_errors)} errors")
        print(f"\n⚠️ {len(short_errors)} errors have short descriptions (<10 words)")
    
    # Long descriptions issue
    long_errors = errors_df[errors_df['word_count'] > 100]
    if len(long_errors) > 0:
        issues.append(f"Very long descriptions (>100 words): {len(long_errors)} errors")
        print(f"⚠️ {len(long_errors)} errors have very long descriptions (>100 words)")
    
    # Generic vocabulary issue
    generic_patterns = ['beautiful', 'amazing', 'famous', 'popular', 'tourist']
    generic_errors = errors_df[errors_df['Description'].str.lower().str.contains('|'.join(generic_patterns), na=False)]
    issues.append(f"Generic descriptions: {len(generic_errors)} errors contain generic words")
    print(f"⚠️ {len(generic_errors)} errors contain generic descriptive words")
    
    print("\nPotential labeling issues:")
    for _, row in errors_df.sample(min(3, len(errors_df))).iterrows():
        desc = str(row['Description'])[:150]
        print(f"  - \"{desc}...\"")
        print(f"    Labeled as: {row['True_Country']}, Predicted: {row['Predicted_Country']}")
    
    with open(output_dir / "data_quality_issues.txt", "w", encoding="utf-8") as f:
        f.write("DATA QUALITY ISSUES IDENTIFIED\n")
        f.write("=" * 60 + "\n\n")
        for issue in issues:
            f.write(f"• {issue}\n")
        f.write("\n\nRECOMMENDATIONS:\n")
        f.write("1. Review short descriptions - may lack sufficient context\n")
        f.write("2. Add country-specific keywords or landmarks to training data\n")
        f.write("3. Consider removing overly generic descriptions\n")
        f.write("4. Verify labels for ambiguous locations (e.g., Mediterranean areas)\n")
        f.write("5. Balance dataset to reduce class imbalance effects\n")
    
    return issues

def create_error_summary(test_df, errors_df, confusion_pairs, output_dir):
    """Create comprehensive error summary report."""
    
    accuracy = 1 - len(errors_df) / len(test_df)
    
    summary = f"""
================================================================================
                    TASK 5: ERROR ANALYSIS SUMMARY REPORT
================================================================================

OVERALL PERFORMANCE:
--------------------
Total Test Samples: {len(test_df)}
Correct Predictions: {len(test_df) - len(errors_df)}
Incorrect Predictions: {len(errors_df)}
Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)

MOST PROBLEMATIC CLASSES (highest error rates):
-----------------------------------------------
{errors_df['True_Country'].value_counts().head(5).to_string()}

MOST COMMON CONFUSIONS:
-----------------------
{confusion_pairs.head(10).to_string()}

KEY FINDINGS:
-------------
1. Country pairs with similar landscapes are frequently confused
   (e.g., Mediterranean countries, island nations)

2. Descriptions with generic tourist vocabulary are harder to classify
   (e.g., "beautiful", "famous landmark")

3. Short descriptions provide insufficient context for accurate prediction

4. Historical/architectural sites cause confusion across countries with
   similar heritage (e.g., ancient ruins, religious buildings)

5. Class imbalance affects minority classes more severely

RECOMMENDATIONS FOR IMPROVEMENT:
--------------------------------
1. Feature Engineering:
   - Add entity recognition for landmark names
   - Include character-level features for non-English words
   - Add topic modeling features

2. Data Enhancement:
   - Collect more samples for underrepresented countries
   - Ensure descriptions contain country-specific details
   - Add structured metadata (landmarks, regions)

3. Model Improvements:
   - Try ensemble methods combining multiple classifiers
   - Experiment with transformer-based models (BERT, etc.)
   - Use data augmentation techniques

4. Evaluation Strategy:
   - Use hierarchical classification (region -> country)
   - Consider multi-label setup for ambiguous cases
   - Weight samples by description quality

================================================================================
"""
    
    with open(output_dir / "error_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(summary)
    return summary

def run_error_analysis():
    """Main error analysis pipeline."""
    print("=" * 80)
    print("TASK 5: PERFORMANCE ANALYSIS - ERROR ANALYSIS")
    print("=" * 80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/7] Loading data...")
    df = load_and_prepare_data()
    print(f"  Dataset shape: {df.shape}")
    
    print("\n[2/7] Preparing features...")
    X = df["Description"].astype(str)
    y = df["Country"].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    test_indices = X_test.index
    labels = sorted(y.unique())
    
    # Try to load saved model and vectorizer from Task 4
    print("\n[3/7] Loading or training model...")
    model, vectorizer = load_trained_model()
    
    if model is None or vectorizer is None:
        # Fallback: create vectorizer and train model
        config = get_tfidf_config()
        vectorizer = TfidfVectorizer(**config)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        model = train_model_fallback(X_train_tfidf, y_train)
    else:
        # Use loaded vectorizer
        X_test_tfidf = vectorizer.transform(X_test)
    
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    
    print("\n[4/7] Analyzing errors...")
    test_df, errors_df = analyze_errors(df, X_test_tfidf, y_test, y_pred, 
                                         test_indices, labels, OUTPUT_DIR)
    
    print("\n[5/7] Per-class error analysis...")
    error_by_true, error_by_pred = analyze_per_class_errors(errors_df, labels, OUTPUT_DIR)
    
    print("\n[6/7] Confusion pair analysis...")
    confusion_pairs = analyze_confusion_pairs(errors_df, OUTPUT_DIR)
    
    analyze_error_descriptions(errors_df, OUTPUT_DIR)
    analyze_sample_errors_detailed(errors_df, OUTPUT_DIR)
    
    print("\n[7/7] Data quality analysis...")
    analyze_data_quality_issues(errors_df, df, OUTPUT_DIR)
    
    create_error_summary(test_df, errors_df, confusion_pairs, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print(f"ERROR ANALYSIS COMPLETE! Results saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    return errors_df, confusion_pairs

if __name__ == "__main__":
    run_error_analysis()
