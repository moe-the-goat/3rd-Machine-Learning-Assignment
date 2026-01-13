# Country Classification from Travel Descriptions

## Project Overview

This machine learning project classifies countries based on textual travel descriptions. The goal is to predict which country a travel description refers to using various ML and deep learning approaches.

## Dataset

- **Source**: `dataset_from_students_only_mode_v1.csv`
- **Samples**: 620 travel descriptions
- **Classes**: 19 countries (after filtering countries with ≥20 samples)
- **Split**: 80% training / 20% testing (stratified)

## Final Model Performance

| Model | Accuracy | Macro F1 | Type |
|-------|----------|----------|------|
| **Transformer (DistilBERT)** | **87.1%** | **85.2%** | Deep Learning |
| **Random Forest** | 83.1% | 80.7% | Traditional ML |
| k-NN (k=3) | 75.8% | 73.0% | Baseline |
| k-NN (k=1) | 74.2% | 71.3% | Baseline |

**Selected Models for Task 4**: Random Forest (best traditional ML) and DistilBERT Transformer (best overall)

## Project Structure

```
ML_Assignmenttemp/
├── code/                          # All Python source files (Tasks 1-5)
│   ├── main_pipeline.py           # Main orchestration script
│   ├── task1_learning_task.py     # Learning task definition
│   ├── task2_eda.py               # Exploratory Data Analysis
│   ├── task3_baseline.py          # k-NN baseline models
│   ├── task4_models.py            # Random Forest + Transformer models
│   ├── task5_error_analysis.py    # Error analysis for RF & Transformer
│   └── utils.py                   # Shared utilities
│
├── Results/                       # All output results organized by task
│   ├── Task1_Output/              # Task definition document
│   ├── Task2_EDA/                 # Exploratory data analysis results
│   ├── Task3_Baseline/            # k-NN baseline results
│   ├── Task4_Models/              # Main model results (RF + Transformer)
│   │   ├── best_model_rf.pkl      # Trained Random Forest model
│   │   ├── best_model_transformer/# Trained DistilBERT checkpoint
│   │   ├── model_comparison.csv   # Performance comparison table
│   │   ├── confusion_*.csv        # Confusion matrices
│   │   └── report_*.txt           # Classification reports
│   └── Task5_ErrorAnalysis/       # Error analysis results
│
├── Other_Models_Tried/            # Alternative models explored
│   ├── code/                      # Code for other models
│   │   ├── svm_model.py
│   │   └── textcnn_model.py
│   ├── results/                   # Results from other models
│   │   ├── SVM/
│   │   ├── TextCNN/
│   └── README.txt
│
├── glove/                         # GloVe embeddings (for TextCNN)
├── dataset_from_students_only_mode_v1.csv
├── requirements.txt
└── README.md
```

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# For transformer model, also install:
pip install torch transformers
```

## Usage

### Run Complete Pipeline

```bash
cd code
python main_pipeline.py
```

### Run Individual Tasks

```bash
cd code

# Task 1: Define learning task
python task1_learning_task.py

# Task 2: Exploratory Data Analysis
python task2_eda.py

# Task 3: k-NN Baselines
python task3_baseline.py

# Task 4: Main Models (Random Forest + Transformer)
python task4_models.py

# Task 5: Error Analysis
python task5_error_analysis.py
```

## Methodology

### Text Preprocessing
- Lowercasing
- Removal of special characters and numbers
- TF-IDF vectorization (max 5000 features)

### Models Implemented

1. **Baselines (Task 3)**:
   - k-NN with k=1 and k=3

2. **Main Models (Task 4)**:
   - **Random Forest**: 200 estimators, max_depth=30, hyperparameter tuned
   - **DistilBERT Transformer**: Fine-tuned for 3 epochs with early stopping

3. **Other Models Tried**:
   - SVM with RBF kernel
   - TextCNN with GloVe embeddings
   - Logistic Regression

### Error Analysis (Task 5)
- Confusion pattern analysis
- Misclassified example examination
- Common error pairs identification (e.g., UK↔Ireland, Italy↔Spain)

## Key Findings

1. **Transformer dominates**: DistilBERT achieves 87.1% accuracy, outperforming all other models
2. **Random Forest is competitive**: Best traditional ML model with 83.1% accuracy
3. **Geographic confusion**: Countries in similar regions are often confused (e.g., European countries)
4. **Class imbalance impact**: Smaller classes (e.g., Australia, Egypt) show lower recall

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- torch (for transformer)
- transformers (for DistilBERT)

## Author

Mohammad