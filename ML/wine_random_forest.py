from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split)
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('Dataset/winequality-red.csv')
df = df.drop_duplicates()

X = df.drop('quality', axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier (
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted: ", y_pred)
print("Actual: ", y_test.values)
print("Accucracy: ", accuracy_score(y_test, y_pred))