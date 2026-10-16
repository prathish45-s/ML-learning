from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (accuracy_score)
import pandas as pd

df = pd.read_csv('Dataset/winequality-red.csv')

df = df.drop_duplicates()

X = df.drop('quality', axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

params = {
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10, 15, 20]
}



model = DecisionTreeClassifier(random_state=42)

grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

print("Best Param: ", grid.best_params_)
print("Best CV Score: ", grid.best_score_)
print("Predict: ", y_pred)
print("Actual: ", y_test.values)
print("Accuracy: ", accuracy_score(y_test, y_pred))