from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('Dataset/Salary_dataset.csv')
df['prediction'] = (df["Salary"] > 80000).astype(int)
X = df[["YearsExperience", "Salary"]]
y = df['prediction']

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print(model.predict(X_test_scaler))