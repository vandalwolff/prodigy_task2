# task2.py
# Customer Segmentation using K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("Mall_Customers.csv")

# -----------------------------
# Select Features
# -----------------------------
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Elbow Method to Find Optimal K
# -----------------------------
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# -----------------------------
# Apply K-Means (K = 5)
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

# -----------------------------
# Visualize Clusters
# -----------------------------
plt.figure()
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.show()

# -----------------------------
# Display Sample Output
# -----------------------------
print(data.head())
