from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

import pandas as pd

df = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Pass":  [0, 0, 0, 0, 1, 1, 1, 1]
})

X = df[["Hours"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)
acc_scr = accuracy_score(y_test, y_pred)
print(acc_scr)