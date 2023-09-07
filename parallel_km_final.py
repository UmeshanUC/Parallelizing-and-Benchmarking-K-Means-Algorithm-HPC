from mpi4py import MPI
import numpy as np
import sys
from helpers.metahelper import save_meta
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def compute_distance(X, centroids):
    distance = np.linalg.norm(X[:, None] - centroids, axis=2)
    return distance


def initialize_clusters(X, K, prev_centroids=None):
    bcast_time = 0
    if prev_centroids is not None:
        centroids = prev_centroids
        return centroids

    centroids = np.empty((K, X.shape[1]), dtype=np.float64)

    if rank == 0:
        centroid_indices = np.random.choice(len(X), K, replace=False)
        centroids = X[centroid_indices.tolist()]

        # Broadcast the centroids to all processes
        bcast_time = bcast_and_measure(centroids)
    else:
        centroids = comm.bcast(centroids, root=0)

    return centroids, bcast_time

def bcast_and_measure(centroids):
    bcast_time = time.time()
    comm.bcast(centroids, root=0)
    return time.time() - bcast_time


def update_clusters(X, labels, K):
    new_centroids = np.empty((K, X.shape[1]), dtype=np.float64)

    for i in range(K):
        # If there are no data points assigned to this centroid, skip it
        if len(X[labels == i]) == 0:
            continue

        local_centroid = np.mean(X[labels == i], axis=0)
        local_centroid, allreduce_time = all_reduce_and_measure(local_centroid)
        new_centroids[i] = local_centroid / size

    return new_centroids, allreduce_time

def all_reduce_and_measure(local_centroid):
    allreduce_time = time.time()
    local_centroid = comm.allreduce(local_centroid, op=MPI.SUM)

    return local_centroid, time.time() - allreduce_time


def parallel_kmeans_clustering(K, D, X, iterations, prev_centroids=None):
    communication_time = 0
    
    centroids, bcast_time = initialize_clusters(X, K, prev_centroids)
    communication_time += bcast_time

    cluster_labels = None


    for _ in range(iterations):
        # Compute the distance between each data point and the centroids
        distance = compute_distance(X, centroids)

        # Find the closest centroid for each data point
        cluster_labels = np.argmin(distance, axis=1)

        # Update the centroids
        centroids, allreduce_time = update_clusters(X, cluster_labels, K)
        communication_time += allreduce_time


    return centroids, cluster_labels,  communication_time


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
    
    K = 3  # Number of clusters
    D = 2  # Dimensionality of data points
    iterations = 1000  # Number of iterations

    data_loading_time = time.time()
    data = load_data(n_samples)
    data_loading_time = time.time() - data_loading_time

    total_time = time.time()
    final_centroids, cluster_labels, communication_time = parallel_kmeans_clustering(
        K, D, data, iterations
    )
    total_time = time.time() - total_time
    computation_time = total_time - communication_time

    if rank == 0:
        # global_comp_time = [0]
        # comm.Reduce(comp_time, global_comp_time, op=MPI.MAX, root=0)
        save_meta([data_loading_time, computation_time, communication_time], "para")


    print("Final centroids:")
    print(final_centroids)
    print("Cluster labels for data points:")
    print(cluster_labels)