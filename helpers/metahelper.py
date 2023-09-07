import csv

# Row format
# row --> data loading time, computation time, communication time

def save_meta(row, type):
    # Open the CSV file in append mode
    filename = "meta_seq.csv"
    
    if type == "para":
        filename ="meta_para.csv"

    with open(filename, 'a') as csvfile:
    # Create a writer object
        writer = csv.writer(csvfile)

        # Write a row to the file
        writer.writerow(row)
