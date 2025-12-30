import os
import re
import glob
import pandas as pd

# ================== CONFIG ==================
BASE = r"C:\Users\Hamed\Desktop\compressed-attachments-from-5748823\compressed-attachments"
OUT_PATH = os.path.join(BASE, "dataset_from_students_only_mode_v1.csv")

ALLOWED = {
    "Weather": {"Sunny", "Rainy", "Cloudy", "Snowy", "Not Clear"},
    "Time of Day": {"Morning", "Afternoon", "Evening"},
    "Season": {"Spring", "Summer", "Fall", "Winter", "Not Clear"},
    "Mood/Emotion": {"Excitement", "Happiness", "Curiosity", "Nostalgia", "Adventure", "Romance", "Melancholy"},
}
STD_COLS = ["Image URL", "Description", "Country", "Weather", "Time of Day", "Season", "Activity", "Mood/Emotion"]

# IMPORTANT: match only student submission files like: 2935882-1190408.csv
STUDENT_FILE_RE = re.compile(r"^\d{7}-\d{7}\s*\.csv$", re.IGNORECASE)

# ================== HELPERS ==================
def normalize_missing(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA, "NaN": pd.NA, "None": pd.NA})

def pick_columns(df: pd.DataFrame, patterns):
    cols = []
    for p in patterns:
        cols += [c for c in df.columns if re.search(p, c, re.IGNORECASE)]
    return list(dict.fromkeys(cols))

def first_non_null(df: pd.DataFrame, cols):
    if not cols:
        return pd.Series([pd.NA] * len(df), dtype="string")
    return df[cols].bfill(axis=1).iloc[:, 0].astype("string")

def canonicalize_weather(x):
    if x is None or pd.isna(x): return pd.NA
    t = str(x).strip().lower()
    if "sunny" in t or t == "clear": return "Sunny"
    if "rain" in t: return "Rainy"
    if "cloud" in t or "overcast" in t: return "Cloudy"
    if "snow" in t: return "Snowy"
    if "not clear" in t or "unclear" in t: return "Not Clear"
    return str(x).strip()

def canonicalize_time(x):
    if x is None or pd.isna(x): return pd.NA
    t = str(x).strip().lower()
    if t in {"morning", "am"}: return "Morning"
    if t in {"afternoon", "pm", "noon"}: return "Afternoon"
    if t in {"evening"}: return "Evening"
    # STRICT: anything else (night/later/late) not allowed
    return pd.NA

def canonicalize_season(x):
    if x is None or pd.isna(x): return pd.NA
    t = str(x).strip().lower()
    if "spring" in t: return "Spring"
    if "summer" in t: return "Summer"
    if "fall" in t or "autumn" in t: return "Fall"
    if "winter" in t: return "Winter"
    if "not clear" in t or "unclear" in t: return "Not Clear"
    return str(x).strip()

def canonicalize_mood(x):
    if x is None or pd.isna(x): return pd.NA
    t = str(x).strip().lower()
    mapping = {
        "excitement": "Excitement",
        "excited": "Excitement",
        "happiness": "Happiness",
        "happy": "Happiness",
        "curiosity": "Curiosity",
        "curious": "Curiosity",
        "nostalgia": "Nostalgia",
        "adventure": "Adventure",
        "romance": "Romance",
        "melancholy": "Melancholy",
    }
    return mapping.get(t, pd.NA)  # STRICT: unknown -> NA

def enforce_allowed(series: pd.Series, allowed_set: set) -> pd.Series:
    s = series.astype("string")
    return s.where(s.isin(list(allowed_set)), pd.NA)

def mode_allowed(series: pd.Series, allowed_set: set):
    s = series.dropna()
    s = s[s.isin(list(allowed_set))]
    return s.mode().iloc[0] if len(s) else pd.NA

def read_csv_safely(path: str):
    encs = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_err = None
    for enc in encs:
        # fast
        try:
            return pd.read_csv(path, encoding=enc, dtype="string"), enc, "c"
        except Exception as e:
            last_err = e
        # fallback for broken structure
        try:
            return pd.read_csv(path, encoding=enc, dtype="string",
                               engine="python", sep=None, on_bad_lines="skip"), enc, "python_skip"
        except Exception as e:
            last_err = e
    raise last_err

# ================== 1) COLLECT ONLY STUDENT FILES ==================
all_csvs = glob.glob(os.path.join(BASE, "**", "*.csv"), recursive=True)
student_csvs = [p for p in all_csvs if STUDENT_FILE_RE.match(os.path.basename(p).strip())]

if not student_csvs:
    raise SystemExit("No student submission CSVs found matching pattern 7digits-7digits.csv")

print("Found student CSVs:", len(student_csvs))

frames = []
bad = []

for path in student_csvs:
    try:
        df, enc, eng = read_csv_safely(path)
        df["source_file"] = os.path.basename(path)
        df["source_folder"] = os.path.basename(os.path.dirname(path))  # student folder id
        df["source_encoding"] = enc
        df["source_engine"] = eng
        frames.append(df)
    except Exception as e:
        bad.append((path, str(e)))
        print("⚠️ Skipped:", path, "|", e)

if bad:
    pd.DataFrame(bad, columns=["file", "error"]).to_csv(
        os.path.join(BASE, "unreadable_student_files.csv"),
        index=False, encoding="utf-8-sig"
    )

raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(dtype="string")
print("Loaded student files:", len(frames))
print("Unreadable student files:", len(bad))
print("Total raw rows:", len(raw))

# ================== 2) STANDARDIZE ==================
raw.columns = [c.strip() for c in raw.columns]

img_cols = pick_columns(raw, ["image", "url", "link"])
desc_cols = pick_columns(raw, ["description", "caption", "text", "comment"])
country_cols = pick_columns(raw, ["country", "location", "place"])
activity_cols = pick_columns(raw, ["activity", r"\btype\b"])
weather_cols = pick_columns(raw, [r"\bweather\b"])
time_cols = pick_columns(raw, ["time of day", "timeofday", "time_day", "time of the day"])
season_cols = pick_columns(raw, [r"\bseason\b"])
mood_cols = pick_columns(raw, ["mood", "emotion", "feeling"])

std = pd.DataFrame()
std["Image URL"] = first_non_null(raw, img_cols)
std["Description"] = first_non_null(raw, desc_cols)
std["Country"] = first_non_null(raw, country_cols)
std["Activity"] = first_non_null(raw, activity_cols)
std["Weather"] = first_non_null(raw, weather_cols)
std["Time of Day"] = first_non_null(raw, time_cols)
std["Season"] = first_non_null(raw, season_cols)
std["Mood/Emotion"] = first_non_null(raw, mood_cols)

# keep trace
for c in ["source_file", "source_folder", "source_encoding", "source_engine"]:
    std[c] = raw[c].astype("string") if c in raw.columns else pd.NA

# normalize
for c in STD_COLS:
    std[c] = normalize_missing(std[c])

# ================== 3) DROP FULLY EMPTY ==================
keep_if_any = ["Image URL", "Description", "Country"]
fully_empty = std[keep_if_any].isna().all(axis=1)

dropped_empty = int(fully_empty.sum())
std = std.loc[~fully_empty].copy()

# ================== 4) CANONICALIZE + STRICT ==================
std["Weather"] = std["Weather"].map(canonicalize_weather)
std["Time of Day"] = std["Time of Day"].map(canonicalize_time)
std["Season"] = std["Season"].map(canonicalize_season)
std["Mood/Emotion"] = std["Mood/Emotion"].map(canonicalize_mood)

for c in ["Weather", "Time of Day", "Season", "Mood/Emotion", "Activity"]:
    std[c] = normalize_missing(std[c])

invalid_counts = {}
for col, allowed_set in ALLOWED.items():
    before = std[col].notna().sum()
    std[col] = enforce_allowed(std[col], allowed_set)
    after = std[col].notna().sum()
    invalid_counts[col] = int(before - after)

# ================== 5) MODE IMPUTE ==================
fallback = {col: mode_allowed(std[col], ALLOWED[col]) for col in ALLOWED}
fallback["Activity"] = std["Activity"].dropna().mode().iloc[0] if std["Activity"].dropna().size else pd.NA

for col in ["Weather", "Time of Day", "Season", "Mood/Emotion", "Activity"]:
    m = std[col].isna()
    if m.any():
        std.loc[m, col] = fallback[col]

# duplicates
before_dups = len(std)
std = std.drop_duplicates()
dropped_dups = before_dups - len(std)

std.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

print("\n✅ DONE (students only)")
print("Saved ->", OUT_PATH)
print("Rows after standardize:", len(std))
print("Dropped fully-empty rows:", dropped_empty)
print("Dropped duplicates:", dropped_dups)

print("\nInvalid values converted to NA (strict):")
for k, v in invalid_counts.items():
    print(f"- {k}: {v}")

print("\nMode fallbacks used:")
for k, v in fallback.items():
    print(f"- {k}: {v}")

if bad:
    print("\n⚠️ unreadable_student_files.csv saved in BASE")
per_file = std.groupby("source_file").size().sort_values()
print("\nBottom 15 files by row count:")
print(per_file.head(15))

print("\nTop 15 files by row count:")
print(per_file.tail(15))

per_file.to_csv(os.path.join(BASE, "rows_per_student_file.csv"), encoding="utf-8-sig")
print("\nSaved rows_per_student_file.csv")
