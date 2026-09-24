import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [40, 45, 52, 60, 65, 72, 78, 85]
})

# Feature and Target
X = df[["Hours"]]
y = df["Marks"]

# Model
model = LinearRegression()
model.fit(X, y)

# Slope/ Weight
slope = model.coef_
print(slope)

# Intercept/ bias
intercept = model.intercept_
print(intercept)

my_predict = model.predict([[7]])
print(my_predict)
