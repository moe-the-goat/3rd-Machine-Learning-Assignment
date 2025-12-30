import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# ===============================
# Dataset Path
# ===============================
DATASET_PATH = r"C:\Users\Hamed\Desktop\compressed-attachments-from-5748823\compressed-attachments\dataset_from_students_only_mode_v1.csv"

# ===============================
# Output
# ===============================
OUTPUT_DIR = "Baseline_country_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CM_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix_knn_k3.png")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv(DATASET_PATH)

# ===============================
# Filter countries (>= 20 samples)
# ===============================
country_counts = df["Country"].value_counts()
valid_countries = country_counts[country_counts >= 20].index
df = df[df["Country"].isin(valid_countries)].reset_index(drop=True)

# ===============================
# Features & Labels
# ===============================
X = df["Description"].astype(str)
y = df["Country"].astype(str)

# ===============================
# Train / Test Split
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
# Train kNN (k=3)
# ===============================
knn = KNeighborsClassifier(
    n_neighbors=3,
    metric="cosine"
)

knn.fit(X_train_tfidf, y_train)
y_pred = knn.predict(X_test_tfidf)

# ===============================
# Confusion Matrix
# ===============================
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

# ===============================
# Plot
# ===============================
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    xticklabels=labels,
    yticklabels=labels,
    annot=False,
    cmap="Blues"
)

plt.xlabel("Predicted Country")
plt.ylabel("True Country")
plt.title("Confusion Matrix — kNN (k=3)")
plt.tight_layout()

# ===============================
# Save
# ===============================
plt.savefig(CM_PATH, dpi=300)
plt.show()

print("Confusion matrix saved to:")
print(CM_PATH)
