from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import (accuracy_score)
import pandas as pd

df = pd.read_csv('Dataset/winequality-red.csv')
df = df.drop_duplicates()

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

params = {
    "C": [0.01, 0.1, 1, 10, 100],
}

model = SVC(kernel="rbf", gamma="scale")

grid = GridSearchCV (
    model,
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train_scaler, y_train)

y_pred = grid.predict(X_test_scaler)
print("Best Paramater: ", grid.best_estimator_)
print("Best Score: ", grid.best_score_)
print("Accuarcy: ", accuracy_score(y_test, y_pred))