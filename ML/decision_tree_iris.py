from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris(as_frame=True)

X = iris.data
y = iris.target

X_train, X_test , y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predicted: ", y_pred)
print("Actual: ", y_test.values)
print("Accuracy: ", accuracy_score(y_test, y_pred))


