from sklearn.cluster import KMeans
import numpy as np

# Load your data (you can use your distributed data loader here)
data = np.loadtxt("concrete.csv", delimiter=',')

# Specify the number of clusters (K)
k = 3

# Create a K-Means model
kmeans = KMeans(n_clusters=k)

# Fit the model to your data
kmeans.fit(data)

# Get the cluster centroids
centroids = kmeans.cluster_centers_

# Get the cluster assignments for each data point
cluster_assignments = kmeans.labels_

# Print the cluster centroids and assignments
print("Cluster Centroids:", centroids)
print("Cluster Assignments:", cluster_assignments)