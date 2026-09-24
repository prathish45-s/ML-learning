import pandas as pd

df = pd.DataFrame({
    "Name": ["Arun", "Bala", "Charan", "Divya"],
    "Age": [20, 22, 21, 23],
    "Marks": [75, 82, 68, 90]
})

max_80 = df[df["Marks"] > 80]
print(max_80)

data = {
    "Name": ["Arun", "Bala", "Charan", "Divya", "Esha"],
    "Age": [20, 22, 21, 23, 25],
    "Marks": [75, 82, 68, 90, 88]
}

df = pd.DataFrame(data)
print(df["Age"].mean())
print(df[df["Marks"] > 80])
print(df[df["Age"] >= 23])
print(df.isnull().sum())