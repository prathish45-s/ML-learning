import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Dataset/Mall_Customers.csv')
df = df[["Annual Income (k$)", "Spending Score (1-100)"]]
df.columns = ["Annual Income", "Spending Score"]

X = df[["Annual Income", "Spending Score"]]

interias = []

for k in range(1, 7):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )
    model.fit(X)
    interias.append(model.inertia_)

plt.plot(range(1, 7), interias, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow")
plt.show()