import glob
import os
from data_loader import group_files

filenames = glob.glob("data_*.csv")

print(group_files(filenames, 4))
