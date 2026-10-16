from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaler, y_train)

y_pred = model.predict(X_test_scaler)

print("Prediction: ", y_pred)
print("Actual: ", y_test.values)

print("Accuracy: ", accuracy_score(y_test, y_pred))

