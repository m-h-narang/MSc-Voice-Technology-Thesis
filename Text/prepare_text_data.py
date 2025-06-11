# prepare_text_data.py

import pandas as pd
import os
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords


os.environ["NLTK_DATA"] = "/home1/s6028608/nltk_data"


nltk.download("punkt", download_dir="/home1/s6028608/nltk_data")
nltk.download("stopwords", download_dir="/home1/s6028608/nltk_data")

# === File paths ===
INPUT_CSV = "../../mustard++_text.csv"
OUTPUT_CSV = "../processed/text_data.csv"

# === Load data ===
df = pd.read_csv(INPUT_CSV)

# === Drop rows with missing sentences or sarcasm label ===
df = df.dropna(subset=["SENTENCE", "Sarcasm"])

# === Label Encoding ===
label_columns = ["Sarcasm", "Sarcasm_Type", "Explicit_Emotion", "Implicit_Emotion"]
label_encoders = {}

for col in label_columns:
    if df[col].isnull().all():
        continue
    le = LabelEncoder()
    df[col] = df[col].fillna("NONE")
    df[col + "_encoded"] = le.fit_transform(df[col])
    label_encoders[col] = le

# === Text Cleaning Function ===
stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and digits
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    # Tokenize
    from nltk.tokenize import wordpunct_tokenize
    tokens = wordpunct_tokenize(text)
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# === Apply cleaning ===
df["SENTENCE_clean"] = df["SENTENCE"].apply(clean_text)

# === Save processed data ===
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Processed data saved to: {OUTPUT_CSV}")