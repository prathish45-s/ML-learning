from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv('Dataset/data.csv', index_col=0)
print(df)

X = df[["X_1", "X_2"]]
y = df["y "]

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

class_report = classification_report(y_test, y_pred)
print(class_report)