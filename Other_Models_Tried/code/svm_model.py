"""
SVM Model - Alternative model tried for country classification.
This model achieved ~83% accuracy but was not selected as the main model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from utils import BASE_DIR, load_and_prepare_data, get_tfidf_config, RANDOM_STATE

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "SVM"


def train_svm():
    """Train SVM with hyperparameter tuning."""
    print("=" * 60)
    print("SVM CLASSIFIER")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_prepare_data()
    X = df["Description"].astype(str)
    y = df["Country"].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # TF-IDF
    config = get_tfidf_config()
    vectorizer = TfidfVectorizer(**config)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    labels = sorted(y.unique())
    
    print(f"Train: {X_train_tfidf.shape}, Test: {X_test_tfidf.shape}")
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced']
    }
    
    svm = SVC(random_state=RANDOM_STATE)
    
    print("\nRunning grid search...")
    start = time.time()
    grid = GridSearchCV(svm, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid.fit(X_train_tfidf, y_train)
    
    print(f"Completed in {time.time() - start:.1f}s")
    print(f"Best params: {grid.best_params_}")
    
    # Evaluate
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    
    # Save results
    report = classification_report(y_test, y_pred, digits=4)
    with open(OUTPUT_DIR / "report_svm.txt", "w", encoding="utf-8") as f:
        f.write("SVM RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Parameters: {grid.best_params_}\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Macro F1: {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(OUTPUT_DIR / "confusion_svm.csv", encoding="utf-8-sig")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return accuracy, macro_f1


if __name__ == "__main__":
    train_svm()
