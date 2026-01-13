"""Task 1: Learning Task Definition - Documents the classification problem."""

from utils import BASE_DIR, load_raw_data, load_and_prepare_data, MIN_SAMPLES

OUTPUT_DIR = BASE_DIR / "Results" / "Task1_Output"


def define_learning_task():
    """Generate task definition document with dataset statistics."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    df = load_raw_data()
    df_filtered = load_and_prepare_data(MIN_SAMPLES)
    
    task_definition = f"""
================================================================================
                    MACHINE LEARNING TASK DEFINITION
================================================================================

TASK: Multi-Class Text Classification - Country Prediction from Description
--------------------------------------------------------------------------------

OBJECTIVE:
Given a textual description of a travel/tourism scene, predict the country 
where the scene is located.

MOTIVATION & CREATIVITY:
1. Real-world Application: Useful for travel content tagging, recommendation 
   systems, and automatic geo-tagging of travel blogs.
2. NLP Challenge: Requires understanding cultural, geographical, and 
   architectural cues embedded in text.
3. Cross-cultural Complexity: Similar landscapes exist across countries 
   (beaches, mountains, historic sites), making pure keyword matching insufficient.
4. Imbalanced Classes: Some countries have more samples than others, 
   requiring careful evaluation metrics.

--------------------------------------------------------------------------------
DATASET CHARACTERISTICS:
--------------------------------------------------------------------------------
Total samples in original dataset: {len(df)}
Total samples after filtering (≥{MIN_SAMPLES} per country): {len(df_filtered)}
Number of target classes: {df_filtered['Country'].nunique()}

Class Distribution:
{df_filtered['Country'].value_counts().to_string()}

Feature: Description (text)
- Mean word count: {df_filtered['Description'].str.split().str.len().mean():.2f}
- Min word count: {df_filtered['Description'].str.split().str.len().min()}
- Max word count: {df_filtered['Description'].str.split().str.len().max()}

--------------------------------------------------------------------------------
EVALUATION STRATEGY:
--------------------------------------------------------------------------------
1. Train/Test Split: 80/20 stratified split (random_state=42)
2. Primary Metrics:
   - Accuracy: Proportion of correct predictions
   - Macro F1-Score: Unweighted mean of F1 across classes (handles imbalance)
3. Secondary Analysis:
   - Per-class precision, recall, F1
   - Confusion matrix for error pattern analysis

--------------------------------------------------------------------------------
MODELS TO EVALUATE:
--------------------------------------------------------------------------------
1. Baseline: k-Nearest Neighbors (k=1, k=3) with TF-IDF + cosine distance
2. Proposed Model 1: Random Forest with TF-IDF features
3. Proposed Model 2: Support Vector Machine (SVM) with TF-IDF features

Hyperparameter Tuning:
- Random Forest: n_estimators (50, 100, 200, 300), max_depth (10, 20, 30, None)
- SVM: C (0.1, 1, 10, 100), kernel (linear, rbf)

================================================================================
"""
    
    output_path = OUTPUT_DIR / "task_definition.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(task_definition)
    
    print(task_definition)
    print(f"\nSaved to: {output_path}")
    return df_filtered


if __name__ == "__main__":
    define_learning_task()
