import pandas as pd

df = pd.DataFrame({
    "Name": ["Arun", "Bala", "Charan", "Divya"],
    "Age": [20, None, 21, 23],
    "Marks": [75, 82, None, 90]
})

print(df)

print(df.isnull())

print(df.isnull().sum())

print(df.dropna())

df["Age"] = df["Age"].fillna(df["Age"].mean())
print(df)

df = pd.DataFrame({
    "Name": ["Arun", "Bala", "Charan", "Divya", "Esha"],
    "Gender": ["M", "M", "M", "F", "F"]
})

print(df["Gender"].value_counts())