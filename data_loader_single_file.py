from mpi4py import MPI
import csv
import numpy as np

def load_data_subset(filename, start_idx, end_idx):
    data_subset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for idx, row in enumerate(reader):
            if start_idx <= idx < end_idx:
                # Convert row values to float as needed
                data_row = [float(value) for value in row]
                data_subset.append(data_row)
    return np.array(data_subset)

def distributed_data_loader(filename):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Count the total number of data points in the file
    with open(filename, 'r') as file:
        total_data_size = sum(1 for _ in file)
    
    # Distribute data ranges to processes
    data_per_process = total_data_size // size
    remainder = total_data_size % size
    start_idx = rank * data_per_process
    end_idx = start_idx + data_per_process

    # Allocate remaining data points to some processes
    if rank < remainder:
        start_idx += rank
        end_idx += rank + 1
    else:
        start_idx += remainder
        end_idx += remainder

    # Load data subset for this process
    data_subset = load_data_subset(filename, start_idx, end_idx)
    
    # Gather data from all processes
    gathered_data = comm.allgather(data_subset)
    
    return gathered_data

if __name__ == "__main__":
    filename = "data.csv"
    
    gathered_data = distributed_data_loader(filename)
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    print(f"Process {rank} data shape:", gathered_data[rank].shape)
