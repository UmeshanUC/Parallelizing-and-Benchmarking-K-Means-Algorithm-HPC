#!/bin/bash

# Specify the Python script you want to run
python_script1="data_loader_single_file.py"
python_script2="data_loader_multiple_files.py"

# Start measuring the time
start_time1=$(date +%s%N)

# Run the Python script
mpirun -n 4 python "$python_script1"

# Calculate the elapsed time
end_time1=$(date +%s%N)
elapsed_time1=$((($end_time1 - $start_time1)/1000000)) # Convert nanoseconds to milliseconds

# Print the elapsed time
echo "Elapsed time for data load from single file: ${elapsed_time1}ms"

# Start measuring the time
start_time2=$(date +%s%N)

# Run the Python script
mpirun -n 4 python "$python_script2"

# Calculate the elapsed time
end_time2=$(date +%s%N)
elapsed_time2=$((($end_time2 - $start_time2)/1000000)) # Convert nanoseconds to milliseconds

# Print the elapsed time
echo "Elapsed time for data load from multiple files: ${elapsed_time2}ms"
