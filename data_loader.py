from mpi4py import MPI
import numpy as np

def load_data_subset(filename, start_idx, end_idx):
    # Load data subset from the file
    # Here, you can use any method to load data from the file
    # For example, using NumPy's loadtxt or load function
    
    # For illustration, generating dummy data
    subset = np.arange(start_idx, end_idx)
    return subset

def distribute_data_to_processes(total_data_size, num_processes):
    data_per_process = total_data_size // num_processes
    remainder = total_data_size % num_processes
    
    data_ranges = []
    start_idx = 0
    for rank in range(num_processes):
        end_idx = start_idx + data_per_process
        if rank < remainder:
            end_idx += 1
        data_ranges.append((start_idx, end_idx))
        start_idx = end_idx
    
    return data_ranges

def distributed_data_loader(filename):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Count the total number of data points in the file
    with open(filename, 'r') as file:
        total_data_size = sum(1 for line in file)
    
    # Distribute data ranges to processes
    data_ranges = distribute_data_to_processes(total_data_size, size)
    
    # Load data subset for this process
    start_idx, end_idx = data_ranges[rank]
    data_subset = load_data_subset(filename, start_idx, end_idx)
    
    # Gather data from all processes
    gathered_data = comm.allgather(data_subset)
    
    return gathered_data

if __name__ == "__main__":
    filename = "data.csv"
    
    gathered_data = distributed_data_loader(filename)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    print(f"Process {rank} data:", gathered_data[rank])
