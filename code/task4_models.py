"""
Task 4: Proposed Machine Learning Models
Implements Random Forest and SVM with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from utils import (
    BASE_DIR, load_and_prepare_data, get_tfidf_config, RANDOM_STATE
)

OUTPUT_DIR = BASE_DIR / "Task4_Models"
np.random.seed(RANDOM_STATE)

def create_features(df):
    """Create train/test split and TF-IDF features."""
    X = df["Description"].astype(str)
    y = df["Country"].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    config = get_tfidf_config()
    vectorizer = TfidfVectorizer(**config)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    labels = sorted(y.unique())
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, labels, vectorizer

def evaluate_model(model, X_test, y_test, labels):
    """Evaluate a trained model."""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred
    }

def train_random_forest(X_train, X_test, y_train, y_test, labels, output_dir):
    """Train and tune Random Forest with grid search."""
    print("\n" + "=" * 60)
    print("MODEL 1: RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }
    
    print("\nHyperparameter Grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    print("\nPerforming Grid Search with 5-fold CV...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"Grid search completed in {elapsed:.1f} seconds")
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (Macro F1): {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    results = evaluate_model(best_model, X_test, y_test, labels)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['weighted_f1']:.4f}")
    
    # Save tuning results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results_summary = cv_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    cv_results_summary = cv_results_summary.sort_values('rank_test_score')
    cv_results_summary.to_csv(output_dir / "rf_hyperparameter_tuning.csv", index=False)
    
    plot_hyperparameter_impact(cv_results, 'n_estimators', output_dir / "rf_n_estimators_impact.png")
    plot_hyperparameter_impact(cv_results, 'max_depth', output_dir / "rf_max_depth_impact.png")
    
    # Save report
    with open(output_dir / "report_random_forest.txt", "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST CLASSIFIER RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Best CV Score: {grid_search.best_score_:.4f}\n\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Test Macro F1: {results['macro_f1']:.4f}\n")
        f.write(f"Test Weighted F1: {results['weighted_f1']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results['report'])
    
    # Save confusion matrix
    cm_df = pd.DataFrame(results['confusion_matrix'], index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_random_forest.csv", encoding="utf-8-sig")
    plot_confusion_matrix(results['confusion_matrix'], labels, 
                          "Confusion Matrix — Random Forest", 
                          output_dir / "confusion_random_forest.png")
    
    # Save model for reuse
    joblib.dump(best_model, output_dir / "best_model_rf.pkl")
    
    return {
        "model_name": "Random Forest",
        "best_params": grid_search.best_params_,
        "cv_score": grid_search.best_score_,
        "test_accuracy": results['accuracy'],
        "test_macro_f1": results['macro_f1'],
        "test_weighted_f1": results['weighted_f1'],
        "classifier": best_model,
        "y_pred": results['y_pred']
    }

def train_svm(X_train, X_test, y_train, y_test, labels, output_dir):
    """Train and tune SVM with grid search."""
    print("\n" + "=" * 60)
    print("MODEL 2: SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 60)
    
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'class_weight': ['balanced']
    }
    
    print("\nHyperparameter Grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    svm_base = SVC(random_state=RANDOM_STATE, cache_size=1000)
    
    print("\nPerforming Grid Search with 5-fold CV...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        svm_base,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"Grid search completed in {elapsed:.1f} seconds")
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score (Macro F1): {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    results = evaluate_model(best_model, X_test, y_test, labels)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['weighted_f1']:.4f}")
    
    # Save tuning results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results_summary = cv_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    cv_results_summary = cv_results_summary.sort_values('rank_test_score')
    cv_results_summary.to_csv(output_dir / "svm_hyperparameter_tuning.csv", index=False)
    
    plot_hyperparameter_impact(cv_results, 'C', output_dir / "svm_C_impact.png")
    
    # Save report
    with open(output_dir / "report_svm.txt", "w", encoding="utf-8") as f:
        f.write("SUPPORT VECTOR MACHINE (SVM) RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Best CV Score: {grid_search.best_score_:.4f}\n\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Test Macro F1: {results['macro_f1']:.4f}\n")
        f.write(f"Test Weighted F1: {results['weighted_f1']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results['report'])
    
    # Save confusion matrix
    cm_df = pd.DataFrame(results['confusion_matrix'], index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_svm.csv", encoding="utf-8-sig")
    plot_confusion_matrix(results['confusion_matrix'], labels,
                          "Confusion Matrix — SVM",
                          output_dir / "confusion_svm.png")
    
    # Save model for reuse in Task 5
    joblib.dump(best_model, output_dir / "best_model_svm.pkl")
    
    return {
        "model_name": "SVM",
        "best_params": grid_search.best_params_,
        "cv_score": grid_search.best_score_,
        "test_accuracy": results['accuracy'],
        "test_macro_f1": results['macro_f1'],
        "test_weighted_f1": results['weighted_f1'],
        "classifier": best_model,
        "y_pred": results['y_pred']
    }

def plot_hyperparameter_impact(cv_results, param_name, output_path):
    """Plot the impact of a hyperparameter on model performance."""
    # Extract relevant data
    df = pd.DataFrame(cv_results)
    
    # Get unique values for this parameter
    param_col = f'param_{param_name}'
    if param_col not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by the parameter
    grouped = df.groupby(param_col)['mean_test_score'].agg(['mean', 'std'])
    
    x_labels = [str(x) for x in grouped.index]
    x_pos = range(len(x_labels))
    
    ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], capsize=5, 
           color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Mean CV Score (Macro F1)', fontsize=12)
    ax.set_title(f'Hyperparameter Impact: {param_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_confusion_matrix(cm, labels, title, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.xlabel("Predicted Country", fontsize=12)
    plt.ylabel("True Country", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_model_comparison(results_list, baseline_results, output_path):
    """Create comparison plot of all models."""
    # Combine baseline and proposed models
    all_results = []
    
    # Add baseline results
    all_results.append({
        'Model': 'kNN (k=1)',
        'Accuracy': 0.7963,  # From previous results
        'Macro F1': 0.7940
    })
    all_results.append({
        'Model': 'kNN (k=3)',
        'Accuracy': 0.7870,
        'Macro F1': 0.7901
    })
    
    # Add proposed models
    for r in results_list:
        all_results.append({
            'Model': r['model_name'],
            'Accuracy': r['test_accuracy'],
            'Macro F1': r['test_macro_f1']
        })
    
    df = pd.DataFrame(all_results)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    colors = ['steelblue', 'steelblue', 'forestgreen', 'coral']
    df.plot(x='Model', y='Accuracy', kind='bar', ax=axes[0], color=colors, 
            edgecolor='black', legend=False)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Comparison: Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0.5, 1.0])
    for i, v in enumerate(df['Accuracy']):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Macro F1 comparison
    df.plot(x='Model', y='Macro F1', kind='bar', ax=axes[1], color=colors,
            edgecolor='black', legend=False)
    axes[1].set_ylabel('Macro F1', fontsize=12)
    axes[1].set_title('Model Comparison: Macro F1', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0.5, 1.0])
    for i, v in enumerate(df['Macro F1']):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return df

def run_models():
    """Main function to train and evaluate all models."""
    print("=" * 80)
    print("TASK 4: PROPOSED MACHINE LEARNING MODELS")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n[1/4] Loading data...")
    df = load_and_prepare_data()
    print(f"  Dataset shape: {df.shape}")
    print(f"  Number of classes: {df['Country'].nunique()}")
    
    print("\n[2/4] Creating features...")
    X_train, X_test, y_train, y_test, labels, vectorizer = create_features(df)
    print(f"  Train size: {X_train.shape[0]}")
    print(f"  Test size: {X_test.shape[0]}")
    print(f"  Feature dimensions: {X_train.shape[1]}")
    
    # Save vectorizer for Task 5 error analysis
    joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.pkl")
    
    print("\n[3/4] Training and tuning models...")
    
    results = []
    
    rf_results = train_random_forest(X_train, X_test, y_train, y_test, labels, OUTPUT_DIR)
    results.append(rf_results)
    
    svm_results = train_svm(X_train, X_test, y_train, y_test, labels, OUTPUT_DIR)
    results.append(svm_results)
    
    print("\n[4/4] Creating comparison...")
    
    comparison_df = plot_model_comparison(results, None, OUTPUT_DIR / "model_comparison.png")
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    
    print("\n" + "=" * 80)
    print("TASK 4 SUMMARY: MODEL COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "-" * 60)
    print("ANALYSIS OF RESULTS:")
    print("-" * 60)
    
    best_idx = comparison_df['Macro F1'].idxmax()
    best_model = comparison_df.loc[best_idx, 'Model']
    best_f1 = comparison_df.loc[best_idx, 'Macro F1']
    
    baseline_f1 = comparison_df[comparison_df['Model'].str.contains('kNN')]['Macro F1'].max()
    improvement = (best_f1 - baseline_f1) / baseline_f1 * 100
    
    print(f"\nBest Model: {best_model}")
    print(f"Best Macro F1: {best_f1:.4f}")
    print(f"Improvement over baseline: {improvement:+.2f}%")
    
    analysis_text = f"""
PERFORMANCE ANALYSIS
====================

Best Performing Model: {best_model}
Best Macro F1 Score: {best_f1:.4f}

Comparison with Baseline:
- kNN (k=1) Macro F1: 0.7940
- kNN (k=3) Macro F1: 0.7901
- Best Model Macro F1: {best_f1:.4f}
- Relative Improvement: {improvement:+.2f}%

Why the proposed models performed {'better' if improvement > 0 else 'similarly'}:

1. Random Forest Benefits:
   - Ensemble method that reduces overfitting
   - Can capture non-linear relationships in TF-IDF space
   - Feature importance can be analyzed
   - Handles class imbalance with balanced weights

2. SVM Benefits:
   - Effective in high-dimensional spaces (TF-IDF has many features)
   - Works well when classes are separable in feature space
   - Regularization parameter C controls model complexity
   - RBF kernel can model complex decision boundaries

Key Observations:
- Text classification tasks often benefit from linear models due to high dimensionality
- Class imbalance (ranging from 22 to 64 samples) affects smaller classes
- Some countries have distinctive vocabulary (landmarks, cultural terms)
- Similar tourist destinations may cause confusion (European countries, island nations)
"""
    
    with open(OUTPUT_DIR / "performance_analysis.txt", "w", encoding="utf-8") as f:
        f.write(analysis_text)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 80)
    
    return results, (X_train, X_test, y_train, y_test, labels)

if __name__ == "__main__":
    run_models()
