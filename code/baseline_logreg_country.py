import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE = r"C:\Users\Hamed\Desktop\compressed-attachments-from-5748823\compressed-attachments"
DATA_PATH = BASE + r"\dataset_standardized_clean.csv"
OUT_DIR = BASE + r"\Baseline_country"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Load + filter
# -----------------------
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", dtype="string")
df = df[["Description", "Country"]].dropna()

country_counts = df["Country"].value_counts()
valid_countries = country_counts[country_counts >= 15].index
df = df[df["Country"].isin(valid_countries)].copy()

print("Rows after filtering:", len(df))
print("Countries:", df["Country"].nunique())

X = df["Description"].astype(str)
y = df["Country"].astype(str)

# -----------------------
# Train / Test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# TF-IDF
# -----------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------
# Logistic Regression
# -----------------------
logreg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

logreg.fit(X_train_vec, y_train)

y_pred = logreg.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

print("\n=== Logistic Regression ===")
print("Accuracy:", round(acc, 4))
print("Macro F1:", round(f1m, 4))

# Save report
rep = classification_report(y_test, y_pred, digits=4)
with open(os.path.join(OUT_DIR, "report_logreg.txt"), "w", encoding="utf-8") as f:
    f.write(rep)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
cm_df = pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique()))
cm_df.to_csv(os.path.join(OUT_DIR, "confusion_logreg.csv"), encoding="utf-8-sig")

# Save summary line
summary_path = os.path.join(OUT_DIR, "baseline_results.csv")
if os.path.exists(summary_path):
    res = pd.read_csv(summary_path)
else:
    res = pd.DataFrame()

res = pd.concat(
    [res,
     pd.DataFrame([{
         "model": "LogisticRegression",
         "accuracy": acc,
         "macro_f1": f1m
     }])],
    ignore_index=True
)

res.to_csv(summary_path, index=False, encoding="utf-8-sig")
print("✅ Logistic Regression results appended.")
