import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------

df = pd.read_csv("dataset/reviews.csv", encoding='latin1')

# Keep only required columns
df = df[['Review', 'Rate']]

# Convert Rate to numeric
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')

# Remove missing values
df.dropna(inplace=True)

# ----------------------------
# 2️⃣ Convert Rating to Sentiment
# ----------------------------

def rating_to_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df['Sentiment'] = df['Rate'].apply(rating_to_sentiment)

# Drop Rate column (optional)
df = df[['Review', 'Sentiment']]

# ----------------------------
# 3️⃣ Text Cleaning
# ----------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['Review'] = df['Review'].apply(clean_text)

# ----------------------------
# 4️⃣ Train Test Split
# ----------------------------

X = df['Review']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # important for balanced classes
)

# ----------------------------
# 5️⃣ TF-IDF Vectorization
# ----------------------------

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------------------------
# 6️⃣ Train Model
# ----------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'  # improves neutral class
)

model.fit(X_train_tfidf, y_train)

# ----------------------------
# 7️⃣ Evaluate Model
# ----------------------------

y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# 8️⃣ Save Model & Vectorizer
# ----------------------------

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")
