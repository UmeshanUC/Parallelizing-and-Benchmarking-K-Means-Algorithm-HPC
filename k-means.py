import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from helpers.metahelper import save_meta

# Function to assign each data point to the nearest centroid
def assign_to_nearest_centroid(X, centroids):
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

# Function to update the centroids based on the assigned points
def update_centroids(X, labels, k):
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(data, k, max_iterations, tolerance, init_centroids):
    for i in range(max_iterations):
        labels = assign_to_nearest_centroid(data, init_centroids)
        new_centroids = update_centroids(data, labels, k)
    
        if np.all(np.abs(new_centroids - init_centroids) < tolerance):
            break
    
        init_centroids = new_centroids
    return init_centroids,labels


def calculate_inertia(X, labels, centroids):
    inertia = 0
    
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        squared_distances = np.sum((cluster_points - centroids[i])**2)
        inertia += squared_distances
    
    return inertia

def load_data(n_sample=1000):
    # Generate random data points for demonstration
    np.random.seed(0)
    return np.random.rand(n_sample, 2)

if __name__ == "__main__":
    # Access command-line arguments
    script_name = sys.argv[0]
    arguments = sys.argv[1:]
    if len(arguments) == 0 :
        raise Exception("no. of samples should be passed in args")

    n_samples = int(arguments[0])
    data_loading_time = time.time()
    data = load_data(n_samples)
    data_loading_time = time.time() - data_loading_time

    # Define the number of clusters (k)
    k = 3

    # Initialize cluster centroids randomly
    init_centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    max_iterations = 100
    tolerance = 1e-4
    centroids = init_centroids

    computation_time = time.time()
    centroids, labels = kmeans(data, k, max_iterations, tolerance, centroids)
    computation_time = time.time() - computation_time

    inertia = calculate_inertia(data, labels=labels, centroids=centroids)

    save_meta([data_loading_time, computation_time, 0], "seq")


    print("Centroids: ", end="")
    print(centroids)

    print("Labels: ", end="")
    print(labels)

    print("Inertia: ", end="")
    print(inertia)


    # Plot the data points and cluster centroids
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
    plt.title(f'K-Means Clustering with k={k}')
    plt.show()
