from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

df = pd.read_csv("Dataset/spam.csv", encoding="latin-1")

df = df[["v1", "v2"]]
df.columns = ["label", "message"]

X = df["message"]
y = df["label"]

vectorizer = CountVectorizer()
X_vectorizer = vectorizer.fit_transform(X)

print(vectorizer.get_feature_names_out())

model = MultinomialNB()
model.fit(X_vectorizer, y)

new_message = ["free prize money"]
new_message_vectorizer = vectorizer.transform(new_message)

prediction = model.predict(new_message_vectorizer)
print(prediction)