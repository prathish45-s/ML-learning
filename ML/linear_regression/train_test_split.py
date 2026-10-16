from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

df = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [40, 45, 52, 60, 65, 72, 78, 85, 90, 95]
})

X = df[["Hours"]]
y = df["Marks"]

# Random split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Fixed split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print(mse)