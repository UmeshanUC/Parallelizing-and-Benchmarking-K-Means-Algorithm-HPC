from mpi4py import MPI
import csv
import numpy as np
import glob

def load_data_subset(filename, start_idx, end_idx, max_rows):
    data_subset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for idx, row in enumerate(reader):
            if start_idx <= idx < end_idx:
                # Convert row values to float as needed
                data_row = [float(value) for value in row]
                data_subset.append(data_row)

    # Pad with zeros if needed to reach max_rows
    while len(data_subset) < max_rows:
        data_subset.append([0.0] * len(data_subset[0]))
    
    return np.array(data_subset)

def distributed_data_loader(filenames):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    gathered_data = []

    for filename in filenames:
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
        max_rows = total_data_size // size  # Determine max rows
        data_subset = load_data_subset(filename, start_idx, end_idx, max_rows)

        # Append data to the list
        gathered_data.append(data_subset)

    return gathered_data

if __name__ == "__main__":
    filenames = glob.glob("data_*.csv")  # List of your data files

    gathered_data = distributed_data_loader(filenames)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Here, gathered_data is a list of NumPy arrays with the same number of rows
    for idx, data in enumerate(gathered_data):
        print(f"Process {rank}, File {idx}, Data shape: {data.shape}")
