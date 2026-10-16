from sklearn.linear_model import Ridge
from sklearn.model_selection import (train_test_split, GridSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error, root_mean_squared_error)
import pandas as pd

df = pd.read_csv('Dataset/winequality-red.csv')
print(df.columns)


print("Is Null value: ",df.isnull().sum())
print("Duplicate sum: ", df.duplicated().sum())

df = df.drop_duplicates()

X = df.drop('quality', axis=1)
y = df['quality']

params = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}

model = Ridge()

grid = GridSearchCV (
    model,
    params, 
    cv=5,
    scoring="r2"
)

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42
)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
grid.fit(X_train_scaler, y_train)

y_pred = grid.predict(X_test_scaler)
print("Best Param: ", grid.best_params_)        
print("Predict: ", y_pred.round())
print("Actual: ", y_test.values)
print("MSE: ", mean_squared_error(y_test, y_pred))
print("RMSE: ", root_mean_squared_error(y_test, y_pred))
print("r2_score: ", r2_score(y_test, y_pred))

