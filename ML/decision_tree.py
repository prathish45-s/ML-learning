from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.DataFrame({
    "Hours":      [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": [60, 65, 70, 72, 80, 85, 90, 95],
    "Pass":       [0, 0, 0, 0, 1, 1, 1, 1]
})

X = df[["Hours", "Attendance"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42
)

model = DecisionTreeClassifier(max_depth=3,
                               min_samples_split=4,
                               min_samples_leaf=2,
                               random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Prediction: ", y_pred)
print("Actual: ", y_test.values)