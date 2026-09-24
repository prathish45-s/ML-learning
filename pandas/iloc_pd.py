import pandas as pd

data = {
    "Name": ["Appa", "Amma", "Prathish", "Darshana"],
    "Age": [45, 40, 23, 20],
    "Marks": [100, 90, 80, 70]
}

df = pd.DataFrame(data)

print(df.iloc[0])

# Select Specific values
print(df.iloc[0, 1])

data = {
    "Name": ["Arun", "Bala", "Charan", "Divya"],
    "Age": [20, 22, 21, 23],
    "Marks": [75, 82, 68, 90]
}

df = pd.DataFrame(data)

print(df["Marks"])
print(df[["Name", "Age"]])
print(df.iloc[2])