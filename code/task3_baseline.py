"""
Task 3: Baseline Model - k-Nearest Neighbors
Evaluates k-NN with TF-IDF features and cosine distance (k=1 and k=3).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils import (
    BASE_DIR, load_and_prepare_data, get_tfidf_config, RANDOM_STATE, create_data_splits
)

OUTPUT_DIR = BASE_DIR / "Results" / "Task3_Baseline"


def create_tfidf_features(X_train, X_test):
    """Create TF-IDF feature vectors."""
    config = get_tfidf_config()
    vectorizer = TfidfVectorizer(**config)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, vectorizer

def evaluate_knn(X_train, X_test, y_train, y_test, k, labels):
    """Train and evaluate k-NN classifier with specified k value."""
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine",
        weights="distance",
        n_jobs=-1
    )
    
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    return {
        "model": f"kNN (k={k})",
        "k": k,
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "y_pred": y_pred,
        "report": report,
        "confusion_matrix": cm,
        "classifier": knn
    }

def plot_confusion_matrix(cm, labels, title, output_path):
    """Generate and save confusion matrix heatmap."""
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel("Predicted Country", fontsize=12)
    plt.ylabel("True Country", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def run_baseline():
    """Main baseline evaluation pipeline."""
    print("=" * 80)
    print("TASK 3: BASELINE MODEL EVALUATION")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n[1/5] Loading data...")
    df = load_and_prepare_data()
    print(f"  Dataset shape: {df.shape}")
    print(f"  Number of classes: {df['Country'].nunique()}")
    
    print("\n[2/5] Creating train/test split (70/10/20)...")
    # Use centralized split - for baseline we combine train+val since no validation needed
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(df, include_validation=True)
    # Combine train and val for baseline (baseline doesn't need validation)
    X_train_combined = pd.concat([X_train, X_val])
    y_train_combined = pd.concat([y_train, y_val])
    print(f"  Train size: {len(X_train_combined)} (train + val combined)")
    print(f"  Test size: {len(X_test)}")
    
    print("\n[3/5] Creating TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(X_train_combined, X_test)
    print(f"  Feature dimensions: {X_train_tfidf.shape[1]}")
    
    # Save vectorizer for later use
    joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.pkl")
    
    labels = sorted(df["Country"].unique())
    
    print("\n[4/5] Evaluating k-NN baselines...")
    results = []
    
    for k in [1, 3]:
        print(f"\n  Training kNN (k={k})...")
        result = evaluate_knn(X_train_tfidf, X_test_tfidf, y_train, y_test, k, labels)
        results.append(result)
        
        print(f"    Accuracy: {result['accuracy']:.4f}")
        print(f"    Macro F1: {result['macro_f1']:.4f}")
        print(f"    Weighted F1: {result['weighted_f1']:.4f}")
        
        # Save classification report
        report_path = OUTPUT_DIR / f"report_knn_k{k}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"k-NN Baseline (k={k})\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Macro F1: {result['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {result['weighted_f1']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(result['report'])
        
        # Save confusion matrix and visualization
        cm_df = pd.DataFrame(result['confusion_matrix'], index=labels, columns=labels)
        cm_df.to_csv(OUTPUT_DIR / f"confusion_knn_k{k}.csv", encoding="utf-8-sig")
        
        plot_confusion_matrix(
            result['confusion_matrix'],
            labels,
            f"Confusion Matrix — kNN (k={k})",
            OUTPUT_DIR / f"confusion_knn_k{k}.png"
        )
        
        # Save model for potential reuse
        joblib.dump(result['classifier'], OUTPUT_DIR / f"knn_k{k}_model.pkl")
    
    print("\n[5/5] Saving results...")
    results_df = pd.DataFrame([
        {
            "model": r["model"],
            "k": r["k"],
            "accuracy": round(r["accuracy"], 4),
            "macro_f1": round(r["macro_f1"], 4),
            "weighted_f1": round(r["weighted_f1"], 4)
        }
        for r in results
    ])
    results_df.to_csv(OUTPUT_DIR / "baseline_results.csv", index=False, encoding="utf-8-sig")
    
    # Summary output
    print("\n" + "=" * 80)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    return results, (X_train_tfidf, X_test_tfidf, y_train, y_test, labels, vectorizer)

if __name__ == "__main__":
    run_baseline()
