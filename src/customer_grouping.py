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
print(df.head())
print(df["Cluster"].value_counts().sort_index())

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

cluster_names = {
    0: "Cluster 0: Average Income, Average Spending",
    1: "Cluster 1: High Income, High Spending",
    2: "Cluster 2: Low Income, High Spending",
    3: "Cluster 3: High Income, Low Spending",
    4: "Cluster 4: Low Income, Low Spending"
}

for cluster, name in cluster_names.items():
    cluster_data = df[df["Cluster"] == cluster]

    plt.scatter(
        cluster_data["Annual Income (k$)"],
        cluster_data["Spending Score (1-100)"],
        s=65,
        label=name
    )
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X",
    s=200,
    c="red",
    label="Centroids"
)
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(
    title="Customer Segments",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9
)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()