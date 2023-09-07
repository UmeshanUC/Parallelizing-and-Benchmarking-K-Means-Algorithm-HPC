import csv


def read_file(filename):
    data = []
    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    return data
