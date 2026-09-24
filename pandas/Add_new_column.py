import pandas as pd


df = pd.DataFrame({
    "Name": ["Arun", "Bala", "Charan", "Divya", "Esha"],
    "Age": [20, 22, 21, 23, 25],
    "Marks": [75, 82, 68, 90, 88]
})

# Adding new column

df["passed"] = df["Marks"] > 40

df["After_5_years"] = df["Age"] + 5

print(df)