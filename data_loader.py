from mpi4py import MPI
import numpy as np
import glob
import os

def load_data_subset(filename):
    # Load data subset from the file
    # Implement your data loading mechanism here
    # For example, using NumPy's loadtxt or load function
    data_subset = np.loadtxt(filename, delimiter=',')
    return data_subset


def group_files(filenames, num_categories):
    # Sort files by size in descending order
    sorted_files = sorted(filenames, key=lambda x: os.path.getsize(x), reverse=True)

    # Initialize categories with empty lists
    categories = [[] for _ in range(num_categories)]

    # Initialize total sizes for each category
    category_sizes = [0] * num_categories

    # Iterate through files and assign each to the category with the smallest total size
    for file_path in sorted_files:
        file_size = os.path.getsize(file_path)
        smallest_category = min(range(num_categories), key=lambda i: category_sizes[i])
        categories[smallest_category].append(file_path)
        category_sizes[smallest_category] += file_size

    return categories

def distributed_data_loader(filenames):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_files = len(filenames)

    # Sending files based on file size can be used to balance the load even better if needed
    files_per_process = num_files // size
    remainder = num_files % size
    start_file_idx = rank * files_per_process + min(rank, remainder)
    end_file_idx = start_file_idx + files_per_process + (1 if rank < remainder else 0)
    
    data_subset = []
    for idx in range(start_file_idx, end_file_idx):
        filename = filenames[idx]
        subset = load_data_subset(filename)
        data_subset.append(subset)
    
    # Gather data from all processes
    gathered_data = comm.allgather(data_subset)
    
    return gathered_data

if __name__ == "__main__":
    filenames = glob.glob("data_*.csv")  # List of your data files
    
    gathered_data = distributed_data_loader(filenames)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    for idx, data in enumerate(gathered_data[rank]):
        print(f"Process {rank}, File {idx}, Data shape: {data.shape}")
