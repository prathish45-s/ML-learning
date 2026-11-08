import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# 1. Select features
features = ["Age", "Annual_Income", "Annual_Spending"]
X = df[features].copy()

# 2. Handle missing values
X = X.dropna()

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train DBSCAN
model = DBSCAN(eps=0.06, min_samples=5)
labels = model.fit_predict(X_scaled)

# 5. Evaluate
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
noise_percent = n_noise / len(labels) * 100

print("Clusters:", n_clusters)
print("Noise points:", n_noise)
print("Noise percentage:", round(noise_percent, 2))

# 6. Calculate silhouette (excluding noise)
mask = labels != -1
valid_labels = labels[mask]

if len(set(valid_labels)) >= 2:
    score = silhouette_score(X_scaled[mask], valid_labels)
    print("Silhouette:", round(score, 4))
else:
    print("Silhouette cannot be calculated.")

# 7. Create cluster profile
result = X.loc[X.index].copy()
result["Cluster"] = labels

profile = result[result["Cluster"] != -1].groupby("Cluster").agg(
    Count=("Cluster", "size"),
    Mean_Age=("Age", "mean"),
    Mean_Income=("Annual_Income", "mean"),
    Mean_Spending=("Annual_Spending", "mean")
)

print(profile)