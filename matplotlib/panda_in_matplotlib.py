import pandas as pd
import matplotlib.pyplot as plt
data = {
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)

# Matplotlib
plt.plot(df["Hours"], df["Marks"])
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("Hours vs Marks")
plt.show()