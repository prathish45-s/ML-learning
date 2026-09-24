from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Pass":  [0, 0, 0, 0, 1, 1, 1, 1]
})

X = df[['Hours']]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prod = model.predict_proba(X_test)

print("Predict Class: ")
print(y_pred)

print("Predicted Probability")
print(y_prod)