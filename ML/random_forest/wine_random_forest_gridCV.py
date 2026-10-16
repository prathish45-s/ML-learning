from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
import pandas as pd


df = pd.read_csv("Dataset/winequality-red.csv")
df = df.drop_duplicates()

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42, stratify=y
)

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10]
}

model = RandomForestClassifier(random_state=42, class_weight="balanced")

grid = GridSearchCV (
    model,
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

print("Best Param: ", grid.best_params_)
print("Best CV Score: ", grid.best_score_)
print("Predict: ", y_pred)
print("Actual: ", y_test.values)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))