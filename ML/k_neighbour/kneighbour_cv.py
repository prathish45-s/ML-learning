from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.metrics import accuracy_score
import pandas as pd


# df = pd.DataFrame({
#     "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
#     "Pass":  [0, 0, 0, 0, 1, 1, 1, 1]
# })

df = pd.read_csv('Dataset/Salary_dataset.csv', index_col=0)
df["predict"] = (df['Salary'] > 80000).astype(int) 
X = df[["YearsExperience", "Salary"]]
y = df["predict"]

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42
)
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

model = KNeighborsClassifier()
params = {
    "n_neighbors": [1, 3, 5, 7, 9]
}
grid = GridSearchCV (
    model,
    params,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)