import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


# 1. Load dataset

df = pd.read_csv("contoh-berita.csv")

# Gabungkan title + text
df['full_text'] = df['title'].astype(str) + " " + df['text'].astype(str)


# 2. Split data

X_train, X_test, y_train, y_test = train_test_split(
    df['full_text'],
    df['label'],
    test_size=0.2,
    random_state=7
)


# 3. TF-IDF

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)


# 4. Passive Aggressive Classifier

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)


# 5. Evaluasi

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print(f"\nAkurasi: {round(score*100, 2)}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))


# 6. Simpan model & vectorizer

joblib.dump(pac, "fake_news_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

print("\nModel dan vectorizer berhasil disimpan sebagai:")
print(" - fake_news_model.pkl")
print(" - tfidf_vectorizer.pkl")
