"""
Task 5: Error Analysis
Analyzes misclassifications for Random Forest and Transformer models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from utils import BASE_DIR, load_and_prepare_data, get_tfidf_config, RANDOM_STATE, create_data_splits

OUTPUT_DIR = BASE_DIR / "Results" / "Task5_ErrorAnalysis"
TASK4_DIR = BASE_DIR / "Results" / "Task4_Models"

# Check for transformer availability
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def load_rf_model():
    """Load Random Forest model and vectorizer from Task 4."""
    model_path = TASK4_DIR / "best_model_rf.pkl"
    vectorizer_path = TASK4_DIR / "tfidf_vectorizer.pkl"
    
    if model_path.exists() and vectorizer_path.exists():
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("  Loaded Random Forest model from Task 4")
        return model, vectorizer
    return None, None


def load_transformer_model():
    """Load fine-tuned transformer from Task 4."""
    model_path = TASK4_DIR / "best_model_transformer"
    
    if not TRANSFORMERS_AVAILABLE:
        return None, None, None
    
    if model_path.exists():
        try:
            tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
            model.eval()
            id2label = model.config.id2label if hasattr(model.config, 'id2label') else None
            print("  Loaded Transformer model from Task 4")
            return model, tokenizer, id2label
        except Exception as e:
            print(f"  Warning: Could not load transformer: {e}")
    return None, None, None


def predict_with_transformer(model, tokenizer, texts, device='cpu'):
    """Get transformer predictions."""
    model.to(device)
    model.eval()
    
    all_preds = []
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encodings = tokenizer(batch, truncation=True, padding=True, 
                                  max_length=128, return_tensors='pt').to(device)
            outputs = model(**encodings)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
    
    return np.array(all_preds)


def analyze_errors(test_df, y_test, y_pred, model_name, output_dir):
    """Analyze misclassified samples for a model."""
    test_df = test_df.copy().reset_index(drop=True)
    test_df['True'] = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else y_test
    test_df['Predicted'] = y_pred
    test_df['Correct'] = test_df['True'] == test_df['Predicted']
    
    errors_df = test_df[~test_df['Correct']]
    
    print(f"\n{model_name} Errors: {len(errors_df)}/{len(test_df)} ({len(errors_df)/len(test_df)*100:.1f}%)")
    
    # Save misclassified examples
    output = errors_df[['Description', 'True', 'Predicted']].copy()
    output['Preview'] = output['Description'].str[:150] + "..."
    output.to_csv(output_dir / f"misclassified_{model_name.lower().replace(' ', '_')}.csv", 
                  index=False, encoding='utf-8-sig')
    
    return errors_df


def analyze_confusion_pairs(errors_df, model_name, output_dir):
    """Find most confused country pairs."""
    if len(errors_df) == 0:
        return None
    
    pairs = errors_df.groupby(['True', 'Predicted']).size().sort_values(ascending=False)
    pairs.to_csv(output_dir / f"confusion_pairs_{model_name.lower().replace(' ', '_')}.csv", 
                 header=['Count'])
    
    print(f"\nTop confused pairs ({model_name}):")
    for (true_c, pred_c), count in pairs.head(5).items():
        print(f"  {true_c} → {pred_c}: {count}")
    
    return pairs


def plot_error_distribution(errors_df, model_name, output_dir):
    """Visualize error distribution by class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Errors by true class
    errors_df['True'].value_counts().plot(kind='barh', ax=axes[0], color='coral')
    axes[0].set_xlabel('Errors')
    axes[0].set_title(f'{model_name}: Errors by True Class', fontweight='bold')
    
    # Errors by predicted class
    errors_df['Predicted'].value_counts().plot(kind='barh', ax=axes[1], color='steelblue')
    axes[1].set_xlabel('False Positives')
    axes[1].set_title(f'{model_name}: False Positives by Class', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"error_distribution_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def create_summary(rf_results, transformer_results, output_dir):
    """Create comparison summary of both models."""
    summary = """
================================================================================
                    ERROR ANALYSIS SUMMARY
================================================================================

"""
    if rf_results:
        summary += f"""RANDOM FOREST:
  Accuracy: {rf_results['accuracy']:.4f}
  Macro F1: {rf_results['macro_f1']:.4f}
  Errors: {rf_results['num_errors']}

"""
    
    if transformer_results:
        summary += f"""TRANSFORMER (DistilBERT):
  Accuracy: {transformer_results['accuracy']:.4f}
  Macro F1: {transformer_results['macro_f1']:.4f}
  Errors: {transformer_results['num_errors']}

"""
    
    if rf_results and transformer_results:
        acc_diff = transformer_results['accuracy'] - rf_results['accuracy']
        f1_diff = transformer_results['macro_f1'] - rf_results['macro_f1']
        err_diff = rf_results['num_errors'] - transformer_results['num_errors']
        
        summary += f"""COMPARISON:
  Accuracy improvement (Transformer vs RF): {acc_diff*100:+.2f}%
  Macro F1 improvement: {f1_diff*100:+.2f}%
  Error reduction: {err_diff} fewer errors

CONCLUSION:
  {"Transformer outperforms Random Forest" if f1_diff > 0 else "Random Forest performs competitively"}
  Both models struggle with similar country pairs (see confusion_pairs files)
"""
    
    with open(output_dir / "error_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(summary)


def run_error_analysis():
    """Main error analysis function."""
    print("=" * 80)
    print("TASK 5: ERROR ANALYSIS")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_and_prepare_data()
    
    # Use centralized split (same as training) - get indices to access test data
    X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = create_data_splits(
        df, include_validation=True, return_indices=True
    )
    
    test_df = df.loc[idx_test]
    labels = sorted(df["Country"].unique())
    
    print(f"  Test set: {len(X_test)} samples")
    
    # Load models
    print("\n[2/5] Loading models...")
    rf_model, vectorizer = load_rf_model()
    transformer_model, tokenizer, id2label = load_transformer_model()
    
    rf_results = None
    transformer_results = None
    
    # Random Forest analysis
    if rf_model and vectorizer:
        print("\n[3/5] Analyzing Random Forest errors...")
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred_rf = rf_model.predict(X_test_tfidf)
        
        acc_rf = accuracy_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf, average='macro')
        
        errors_rf = analyze_errors(test_df, y_test, y_pred_rf, "Random_Forest", OUTPUT_DIR)
        analyze_confusion_pairs(errors_rf, "Random_Forest", OUTPUT_DIR)
        plot_error_distribution(errors_rf, "Random Forest", OUTPUT_DIR)
        
        rf_results = {
            'accuracy': acc_rf, 'macro_f1': f1_rf, 'num_errors': len(errors_rf)
        }
    else:
        print("\n[3/5] Random Forest model not found, skipping...")
    
    # Transformer analysis
    if transformer_model and tokenizer:
        print("\n[4/5] Analyzing Transformer errors...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        y_pred_ids = predict_with_transformer(transformer_model, tokenizer, X_test.tolist(), device)
        
        if id2label:
            y_pred_t = [id2label.get(str(i), id2label.get(i, labels[i])) for i in y_pred_ids]
        else:
            y_pred_t = [labels[i] for i in y_pred_ids]
        
        acc_t = accuracy_score(y_test, y_pred_t)
        f1_t = f1_score(y_test, y_pred_t, average='macro')
        
        errors_t = analyze_errors(test_df, y_test, y_pred_t, "Transformer", OUTPUT_DIR)
        analyze_confusion_pairs(errors_t, "Transformer", OUTPUT_DIR)
        plot_error_distribution(errors_t, "Transformer", OUTPUT_DIR)
        
        transformer_results = {
            'accuracy': acc_t, 'macro_f1': f1_t, 'num_errors': len(errors_t)
        }
    else:
        print("\n[4/5] Transformer model not found, skipping...")
    
    # Create summary
    print("\n[5/5] Creating summary...")
    create_summary(rf_results, transformer_results, OUTPUT_DIR)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_error_analysis()
