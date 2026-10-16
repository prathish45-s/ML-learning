from sklearn.linear_model import Ridge
from sklearn.model_selection import (train_test_split, GridSearchCV)
import pandas as pd

df = pd.read_csv('Dataset/Salary_dataset.csv', index_col=0)

X = df[["YearsExperience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Ridge()

params = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

# Select the best model
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
