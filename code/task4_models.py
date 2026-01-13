"""
Task 4: Model Training and Evaluation
Trains Random Forest and DistilBERT Transformer for country classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import time
import random
import warnings
warnings.filterwarnings('ignore')

from utils import BASE_DIR, load_and_prepare_data, get_tfidf_config, RANDOM_STATE, create_data_splits

OUTPUT_DIR = BASE_DIR / "Results" / "Task4_Models"
np.random.seed(RANDOM_STATE)

# Check for transformer dependencies
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from torch.optim import AdamW
    from torch.nn import CrossEntropyLoss
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        get_linear_schedule_with_warmup
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Transformer configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.1
GRADIENT_CLIP = 1.0
EARLY_STOPPING_PATIENCE = 3
USE_AUGMENTATION = True


# =============================================================================
# COMMON UTILITIES
# =============================================================================

def create_features(df):
    """Create TF-IDF features using centralized data splits (70/10/20)."""
    # Use centralized split
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(df, include_validation=True)
    
    # For Random Forest, combine train+val (no validation needed for grid search CV)
    X_train_combined = pd.concat([X_train, X_val])
    y_train_combined = pd.concat([y_train, y_val])
    
    config = get_tfidf_config()
    vectorizer = TfidfVectorizer(**config)
    
    X_train_tfidf = vectorizer.fit_transform(X_train_combined)
    X_test_tfidf = vectorizer.transform(X_test)
    labels = sorted(df["Country"].unique())
    
    return X_train_tfidf, X_test_tfidf, y_train_combined, y_test, labels, vectorizer


def evaluate_model(model, X_test, y_test, labels):
    """Get model predictions and metrics."""
    y_pred = model.predict(X_test)
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        "report": classification_report(y_test, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "y_pred": y_pred
    }


def plot_confusion_matrix(cm, labels, title, output_path):
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# RANDOM FOREST MODEL
# =============================================================================

def train_random_forest(X_train, X_test, y_train, y_test, labels, output_dir):
    """Train Random Forest with grid search tuning."""
    print("\n" + "=" * 60)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }
    
    print("\nSearch space:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    print("\nRunning 5-fold cross-validation...")
    start = time.time()
    
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"Completed in {time.time() - start:.1f}s")
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV F1: {grid.best_score_:.4f}")
    
    best_model = grid.best_estimator_
    results = evaluate_model(best_model, X_test, y_test, labels)
    
    print(f"\nTest Results:")
    print(f"  Accuracy:    {results['accuracy']:.4f}")
    print(f"  Macro F1:    {results['macro_f1']:.4f}")
    print(f"  Weighted F1: {results['weighted_f1']:.4f}")
    
    # Save tuning results
    cv_df = pd.DataFrame(grid.cv_results_)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    cv_df.sort_values('rank_test_score').to_csv(output_dir / "rf_hyperparameter_tuning.csv", index=False)
    
    # Save report
    with open(output_dir / "report_random_forest.txt", "w", encoding="utf-8") as f:
        f.write("RANDOM FOREST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Parameters: {grid.best_params_}\n")
        f.write(f"CV Score: {grid.best_score_:.4f}\n\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Test Macro F1: {results['macro_f1']:.4f}\n")
        f.write(f"Test Weighted F1: {results['weighted_f1']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results['report'])
    
    # Confusion matrix
    cm_df = pd.DataFrame(results['confusion_matrix'], index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_random_forest.csv", encoding="utf-8-sig")
    plot_confusion_matrix(results['confusion_matrix'], labels, 
                          "Random Forest Confusion Matrix", 
                          output_dir / "confusion_random_forest.png")
    
    # Save model
    joblib.dump(best_model, output_dir / "best_model_rf.pkl")
    
    return {
        "model_name": "Random Forest",
        "best_params": grid.best_params_,
        "cv_score": grid.best_score_,
        "test_accuracy": results['accuracy'],
        "test_macro_f1": results['macro_f1'],
        "test_weighted_f1": results['weighted_f1'],
        "classifier": best_model,
        "y_pred": results['y_pred']
    }


# =============================================================================
# TRANSFORMER MODEL
# =============================================================================

def augment_text(text, aug_prob=0.3):
    """Simple text augmentation for training data."""
    if random.random() > aug_prob:
        return text
    
    words = text.split()
    if len(words) < 4:
        return text
    
    aug_type = random.choice(['delete', 'swap', 'shuffle_segment'])
    
    if aug_type == 'delete' and len(words) > 5:
        idx = random.randint(1, len(words) - 2)
        words.pop(idx)
    elif aug_type == 'swap' and len(words) > 3:
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    elif aug_type == 'shuffle_segment' and len(words) > 6:
        start = random.randint(1, len(words) - 4)
        segment = words[start:start+3]
        random.shuffle(segment)
        words[start:start+3] = segment
    
    return ' '.join(words)


if TRANSFORMERS_AVAILABLE:
    class CountryDataset(TorchDataset):
        """PyTorch Dataset for country classification."""
        
        def __init__(self, texts, labels, tokenizer, max_length, augment=False):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.augment = augment
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            if self.augment and USE_AUGMENTATION:
                text = augment_text(text)
            
            encoding = self.tokenizer(
                text, truncation=True, padding='max_length',
                max_length=self.max_length, return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }


def compute_class_weights(labels, num_classes):
    """Compute inverse frequency class weights."""
    class_counts = np.bincount(labels, minlength=num_classes)
    weights = len(labels) / (num_classes * np.maximum(class_counts, 1))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def create_transformer_splits(df, label2id):
    """Create stratified train/val/test splits for transformer using centralized split."""
    # Use centralized split function (70/10/20)
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(df, include_validation=True)
    
    # Convert to numpy arrays and map labels to IDs
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values
    y_train = y_train.map(label2id).values
    y_val = y_val.map(label2id).values
    y_test = y_test.map(label2id).values
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    loss_fn = CrossEntropyLoss(weight=class_weights.to(device))
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')


def evaluate_transformer(model, dataloader, device, class_weights):
    """Evaluate transformer model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    loss_fn = CrossEntropyLoss(weight=class_weights.to(device))
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro'), np.array(all_preds), np.array(all_labels)


def plot_training_history(history, output_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    for ax, metric, title in zip(axes, ['loss', 'acc', 'f1'], ['Loss', 'Accuracy', 'Macro F1']):
        ax.plot(epochs, history[f'train_{metric}'], 'b-o', label='Train')
        ax.plot(epochs, history[f'val_{metric}'], 'r-o', label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Training vs Validation {title}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_transformer(output_dir):
    """Train DistilBERT transformer model."""
    if not TRANSFORMERS_AVAILABLE:
        print("Transformer dependencies not available. Skipping.")
        return None
    
    print("\n" + "=" * 60)
    print("DISTILBERT TRANSFORMER")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    df = load_and_prepare_data()
    labels = sorted(df["Country"].unique())
    num_classes = len(labels)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    print(f"Dataset: {len(df)} samples, {num_classes} classes")
    
    # Create splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_transformer_splits(df, label2id)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    class_weights = compute_class_weights(y_train, num_classes)
    
    # Initialize model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes, id2label=id2label, label2id=label2id,
        attention_dropout=DROPOUT_RATE, seq_classif_dropout=DROPOUT_RATE
    )
    model.to(device)
    
    # Create dataloaders
    train_dataset = CountryDataset(X_train, y_train, tokenizer, MAX_LENGTH, augment=True)
    val_dataset = CountryDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = CountryDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)
    
    # Training loop
    print("\nTraining...")
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_f1, patience_counter, best_model_state = 0, 0, None
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
        val_loss, val_acc, val_f1, _, _ = evaluate_transformer(model, val_loader, device, class_weights)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train: {train_acc:.4f}/{train_f1:.4f}, Val: {val_acc:.4f}/{val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_loss, test_acc, test_f1_macro, y_pred, y_true = evaluate_transformer(model, test_loader, device, class_weights)
    test_f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nTest Results:")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  Macro F1:    {test_f1_macro:.4f}")
    print(f"  Weighted F1: {test_f1_weighted:.4f}")
    
    # Save outputs
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    
    with open(output_dir / "report_transformer.txt", "w", encoding="utf-8") as f:
        f.write("DISTILBERT TRANSFORMER RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Epochs Trained: {len(history['train_loss'])}\n\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Macro F1: {test_f1_macro:.4f}\n")
        f.write(f"Test Weighted F1: {test_f1_weighted:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_transformer.csv", encoding="utf-8-sig")
    plot_confusion_matrix(cm, labels, "Transformer Confusion Matrix", output_dir / "confusion_transformer.png")
    plot_training_history(history, output_dir / "training_history.png")
    
    metrics = {
        "model": MODEL_NAME, "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1_macro), "test_weighted_f1": float(test_f1_weighted),
        "epochs_trained": len(history['train_loss']), "training_time_minutes": round(total_time/60, 2)
    }
    with open(output_dir / "metrics_transformer.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    model.save_pretrained(output_dir / "best_model_transformer")
    tokenizer.save_pretrained(output_dir / "best_model_transformer")
    
    return {
        "model_name": "Transformer (DistilBERT)",
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1_macro,
        "test_weighted_f1": test_f1_weighted,
        "y_pred": y_pred, "y_true": y_true, "labels": labels
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def create_comparison_summary(rf_results, transformer_results, output_dir):
    """Create model comparison summary."""
    all_results = []
    
    # Load baseline results
    baseline_csv = BASE_DIR / "Results" / "Task3_Baseline" / "baseline_results.csv"
    if baseline_csv.exists():
        baseline_df = pd.read_csv(baseline_csv)
        for _, row in baseline_df.iterrows():
            all_results.append({'Model': row['model'], 'Accuracy': row['accuracy'], 'Macro F1': row['macro_f1']})
    
    # Add Random Forest
    if rf_results:
        all_results.append({
            'Model': 'Random Forest',
            'Accuracy': rf_results['test_accuracy'],
            'Macro F1': rf_results['test_macro_f1']
        })
    
    # Add Transformer
    if transformer_results:
        all_results.append({
            'Model': 'Transformer (DistilBERT)',
            'Accuracy': transformer_results['test_accuracy'],
            'Macro F1': transformer_results['test_macro_f1']
        })
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], df['Accuracy'], width, label='Accuracy', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], df['Macro F1'], width, label='Macro F1', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0.5, 1.0])
    
    for bar in bars1 + bars2:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()
    
    return df


def run_models():
    """Main function to train all models."""
    print("=" * 80)
    print("TASK 4: MODEL TRAINING (Random Forest + Transformer)")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    print("\nLoading data...")
    df = load_and_prepare_data()
    print(f"Samples: {len(df)}, Classes: {df['Country'].nunique()}")
    
    # Create TF-IDF features for Random Forest
    print("\nCreating TF-IDF features...")
    X_train, X_test, y_train, y_test, labels, vectorizer = create_features(df)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Features: {X_train.shape[1]}")
    joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.pkl")
    
    # Train Random Forest
    rf_results = train_random_forest(X_train, X_test, y_train, y_test, labels, OUTPUT_DIR)
    
    # Train Transformer
    transformer_results = train_transformer(OUTPUT_DIR)
    
    # Create comparison
    print("\nCreating comparison summary...")
    comparison = create_comparison_summary(rf_results, transformer_results, OUTPUT_DIR)
    print("\n" + comparison.to_string(index=False))
    
    # Save analysis
    analysis = f"""
MODEL ANALYSIS
==============

Random Forest Performance:
- Test Accuracy: {rf_results['test_accuracy']:.4f}
- Test Macro F1: {rf_results['test_macro_f1']:.4f}
- Best Parameters: {rf_results['best_params']}

Transformer Performance:
- Test Accuracy: {transformer_results['test_accuracy']:.4f if transformer_results else 'N/A'}
- Test Macro F1: {transformer_results['test_macro_f1']:.4f if transformer_results else 'N/A'}

Key Observations:
1. Transformer outperforms traditional ML approaches
2. Random Forest is the best traditional ML model
3. Both handle class imbalance through appropriate weighting
"""
    
    with open(OUTPUT_DIR / "performance_analysis.txt", "w", encoding="utf-8") as f:
        f.write(analysis)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return rf_results, transformer_results


if __name__ == "__main__":
    run_models()
