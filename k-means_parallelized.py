from mpi4py import MPI
import numpy as np
import glob
from sklearn.cluster import KMeans


# Function to load data from multiple files and distribute it among processes
k = 3   # No. of clusters
kmeans = KMeans(n_clusters=k)


def distributed_data_loader(filenames):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_files = len(filenames)
    files_per_process = num_files // size
    remainder = num_files % size
    start_file_idx = rank * files_per_process + min(rank, remainder)
    end_file_idx = start_file_idx + files_per_process + \
        (1 if rank < remainder else 0)

    data_subset = []
    for idx in range(start_file_idx, end_file_idx):
        filename = filenames[idx]
        try:
            # Load data from CSV files
            data = np.loadtxt(filename, delimiter=',')
            data_subset.append(data)
        except Exception as e:
            # Handle any exceptions during data loading
            print(f"Error loading data from {filename}: {str(e)}")

    # Gather data from all processes
    gathered_data = comm.allgather(data_subset)

    return gathered_data

# Function to initialize centroids (randomly or using k-means++)


def initialize_centroids(data, k):
    # Implement your initialization logic here
    # You can choose to initialize centroids randomly or using k-means++
    pass

# Function to assign data points to the nearest centroid


def assign_to_centroids(data, centroids):
    # Calculate the Euclidean distance between data points and centroids
    # Assign each data point to the nearest centroid
    pass

# Function to update centroids by averaging data points


def kmeans_fit(data):
    kmeans.fit(data)

# Function to check convergence (e.g., change in centroid positions)


def has_converged(new_centroids, old_centroids, tolerance):
    # Check if the centroids have converged based on a tolerance value
    pass

# Main K-Means algorithm


def parallel_kmeans(data, k, max_iterations, tolerance):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(data)

    # Initialize centroids on one process (e.g., rank 0)
    if rank == 0:
        centroids = initialize_centroids(data, k)
    else:
        centroids = None

    for iteration in range(max_iterations):
        # Broadcast centroids to all processes
        centroids = comm.bcast(centroids, root=0)

        # Update centroids by averaging data points
        new_centroids = kmeans_fit(data)

        # Check for convergence
        if has_converged(new_centroids, centroids, tolerance):
            break

        # Gather new centroids from all processes to update globally
        new_centroids = comm.gather(new_centroids, root=0)

        # On the root process, calculate the global centroids
        if rank == 0:
            print(new_centroids)
            global_centroids = np.mean(new_centroids, axis=0)
        else:
            global_centroids = None

        # Broadcast global centroids to all processes
        global_centroids = comm.bcast(global_centroids, root=0)

        # Use the global centroids for the next iteration
        centroids = global_centroids

    return centroids


if __name__ == "__main__":
    filenames = glob.glob("data_*.csv")  # List of your data files

    gathered_data = distributed_data_loader(filenames)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Combine data from all processes into a single dataset
        data = np.concatenate(
            [subset for sublist in gathered_data for subset in sublist], axis=0)

        # Specify the number of clusters (k), max iterations, and convergence tolerance
        k = 3
        max_iterations = 10
        tolerance = 1

        # Run K-Means
        centroids = parallel_kmeans(data, k, max_iterations, tolerance)

        print("Final centroids:", centroids)
