import os
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIG =========
DATA_PATH = r"C:\Users\Hamed\Desktop\compressed-attachments-from-5748823\compressed-attachments\dataset_from_students_only_mode_v1.csv"
OUT_DIR = r"C:\Users\Hamed\Desktop\compressed-attachments-from-5748823\compressed-attachments\EDA_country_final"

os.makedirs(OUT_DIR, exist_ok=True)

# ========= LOAD =========
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", dtype="string")

# ========= CLEANUP =========
# Clean Description
df["Description"] = df["Description"].astype("string").fillna("").str.strip()
df["Description"] = df["Description"].str.replace(r"\s+", " ", regex=True)

# Clean Country (fix london/London and similar)
# 1) normalize whitespace
# 2) normalize case using casefold
df["Country"] = df["Country"].astype("string").fillna("").str.strip()
df["Country"] = df["Country"].str.replace(r"\s+", " ", regex=True)
df["Country_norm"] = df["Country"].str.casefold()   # normalized for counting
df["Country_display"] = df["Country_norm"].str.title()  # pretty for saving/plots

# ========= SUMMARY STATS =========
summary_lines = []
summary_lines.append(f"Rows: {len(df)}")
summary_lines.append(f"Columns: {len(df.columns)}")

missing_country = (df["Country_norm"].isna() | (df["Country_norm"].str.len() == 0)).sum()
missing_desc = (df["Description"].isna() | (df["Description"].str.len() == 0)).sum()

summary_lines.append(f"Missing Country: {int(missing_country)}")
summary_lines.append(f"Missing Description: {int(missing_desc)}")
summary_lines.append(f"Unique countries: {df['Country_display'].replace('', pd.NA).nunique(dropna=True)}")

# Description length (words)
desc_len_words = df["Description"].fillna("").astype(str).apply(lambda x: len(x.split()))
summary_lines.append(
    f"Description length (words): mean={desc_len_words.mean():.2f}, median={desc_len_words.median():.2f}, max={desc_len_words.max()}"
)

with open(os.path.join(OUT_DIR, "dataset_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

# ========= COUNTRY DISTRIBUTION =========
country_series = df["Country_display"].replace("", pd.NA).dropna()
country_counts = country_series.value_counts()

country_counts.to_csv(
    os.path.join(OUT_DIR, "country_counts_all.csv"),
    header=["count"],
    encoding="utf-8-sig"
)

# Filter countries with >= 20 samples (for modeling)
filtered_countries = country_counts[country_counts >= 20]
filtered_countries.to_csv(
    os.path.join(OUT_DIR, "country_counts_filtered_ge20.csv"),
    header=["count"],
    encoding="utf-8-sig"
)

# ========= PLOTS =========

# 1) Top 15 countries (barh)
top15 = country_counts.head(15).sort_values()
plt.figure()
top15.plot(kind="barh")
plt.title("Top 15 Countries (Count)")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top15_countries.png"), dpi=200)
plt.close()

# 2) Long-tail / class imbalance plot (all countries)
plt.figure()
country_counts.reset_index(drop=True).plot(kind="line")
plt.title("Country Frequency (Long-tail Distribution)")
plt.xlabel("Country rank (sorted by frequency)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_imbalance_plot.png"), dpi=200)
plt.close()

# 3) Description length histogram
plt.figure()
plt.hist(desc_len_words, bins=30)
plt.title("Description Length Distribution (Words)")
plt.xlabel("Words per description")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "description_length.png"), dpi=200)
plt.close()

print("✅ EDA complete.")
print("Saved to:", OUT_DIR)
print("Rows:", len(df), "| Unique countries:", country_counts.shape[0])
print("Countries with >=20 samples:", int((country_counts >= 20).sum()))
