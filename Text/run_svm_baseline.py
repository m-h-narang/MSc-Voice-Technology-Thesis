import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import re

# === Load CSV ===
csv_path = "/scratch/s6028608/MUStARD_Plus_Plus/mustard++_text.csv"
df = pd.read_csv(csv_path)

# Adjust column names
text_col = "SENTENCE"
label_col = "Sarcasm"

# Drop rows with missing labels or text
df = df.dropna(subset=[text_col, label_col])

# Clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower().strip()

df[text_col] = df[text_col].astype(str).apply(clean_text)
df[label_col] = df[label_col].astype(int)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(df[text_col], df[label_col], test_size=0.2, random_state=42)

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === Train SVM ===
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)

# === Evaluation ===
y_pred = svm.predict(X_test_tfidf)

print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Sarcastic", "Sarcastic"]))