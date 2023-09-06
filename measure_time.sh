#!/bin/bash

# Specify the Python script you want to run
python_script="data_loader_multiple_files.py"

# Start measuring the time
start_time=$(date +%s%N)

# Run the Python script
mpirun -n 4 python "$python_script"

# Calculate the elapsed time
end_time=$(date +%s%N)
elapsed_time=$((($end_time - $start_time)/1000000)) # Convert nanoseconds to milliseconds

# Print the elapsed time
echo "Elapsed time: ${elapsed_time}ms"
