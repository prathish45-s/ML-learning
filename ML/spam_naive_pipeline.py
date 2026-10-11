from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report)
import pandas as pd

df = pd.read_csv('Dataset/spam.csv', encoding='latin-1')
df = df[["v1", "v2"]]
df.columns = ['label', 'message']

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifer", MultinomialNB())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))
