import pandas as pd

df = pd.DataFrame({
    "Name": ["Appa", "Amma", "Prathish"],
    "Result": ["Pass", "Fail", "Pass"]
})

df["Result"] = df["Result"].map({
    "Pass": 1,
    "Fail": 0
})

# df["Purchased"] = df["Result"] == "Pass"

df = pd.DataFrame({
    "Name": ["Prathish", "Amma", "Appa"],
    "City": ["Chennai", "Delhi", "Mumbai"]
})

one_hot_emb = pd.get_dummies(df["City"])

print(df)

print(one_hot_emb)