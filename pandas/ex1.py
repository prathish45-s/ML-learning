import pandas as pd

df = pd.DataFrame({
    "Name": ["Arun", "Bala", "Charan", "Divya"],
    "Age": [20, None, 21, 23],
    "Marks": [75, 82, 68, 90]
})

print(df.sort_values("Marks", ascending=False))

df["Age"] = df["Age"].fillna(df["Age"].mean())
print(df)