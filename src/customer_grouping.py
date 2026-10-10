import pandas as pd
df=pd.read_csv("dataset/Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

#finding the optimal number of clusters using the elbow method
from sklearn.cluster import KMeans
inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)
#we used matplotlibe to see the graph and choose k value(here it is 5)

model = KMeans(n_clusters=5, random_state=42, n_init=10)
model.fit(X)

labels = model.labels_
df["Cluster"] = model.labels_

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=65,
    alpha=0.8,
    label="Customers"
)

plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=180,
    c="red",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.legend(
    title="Legend",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()