import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ===============================
# Dataset Path
# ===============================
DATASET_PATH = r"C:\Users\Hamed\Desktop\compressed-attachments-from-5748823\compressed-attachments\dataset_from_students_only_mode_v1.csv"

# ===============================
# Output
# ===============================
OUTPUT_DIR = "Baseline_country_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results_logreg.csv")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv(DATASET_PATH)

print("Original shape:", df.shape)

# ===============================
# Filter countries (>= 20 samples)
# ===============================
country_counts = df["Country"].value_counts()
valid_countries = country_counts[country_counts >= 20].index
df = df[df["Country"].isin(valid_countries)].reset_index(drop=True)

print("Filtered shape:", df.shape)
print("Number of classes:", df["Country"].nunique())

# ===============================
# Features & Labels
# ===============================
X = df["Description"].astype(str)
y = df["Country"].astype(str)

# ===============================
# Train / Test Split (Stratified)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# TF-IDF
# ===============================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    max_df=0.9
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ===============================
# Logistic Regression
# ===============================
logreg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

logreg.fit(X_train_tfidf, y_train)
y_pred = logreg.predict(X_test_tfidf)

# ===============================
# Evaluation
# ===============================
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("\nLogistic Regression Results")
print(f"Accuracy : {acc:.4f}")
print(f"Macro F1 : {f1:.4f}")

# ===============================
# Save Results
# ===============================
results_df = pd.DataFrame([{
    "Model": "Logistic Regression (balanced)",
    "Accuracy": round(acc, 4),
    "Macro_F1": round(f1, 4)
}])

results_df.to_csv(OUTPUT_FILE, index=False)

print("\nResults saved to:")
print(OUTPUT_FILE)
