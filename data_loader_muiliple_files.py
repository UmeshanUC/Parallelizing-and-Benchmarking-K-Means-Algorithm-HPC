from mpi4py import MPI
import numpy as np
import glob

def load_data_subset(filename):
    # Load data subset from the file
    # Implement your data loading mechanism here
    # For example, using NumPy's loadtxt or load function
    data_subset = np.loadtxt(filename, delimiter=',')
    return data_subset

def distributed_data_loader(filenames):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_files = len(filenames)
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
