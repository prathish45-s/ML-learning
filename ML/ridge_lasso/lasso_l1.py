from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score)
import pandas as pd

# 1. Dataset
df = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [40, 45, 52, 60, 65, 72, 78, 85, 90, 95]
})

X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.2, random_state=42
)

model = Lasso(alpha=1.0)
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# 8. Print results
print("Weight:", model.coef_[0])
print("Bias:", model.intercept_)

print("Actual:", y_test.values)
print("Predicted:", y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)