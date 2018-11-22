import pandas as pd
import numpy as np
import os
from UShapelet import get_ushapelet
FILE_NAME = "FourClasses.txt"

file_path = os.path.join(os.getcwd(),FILE_NAME)

# Read the file in from CSV
raw_data = pd.read_csv(file_path,header=None)

# Split the data into a label set (the first column of the data),
# and the actual data, (the second -> N columns).
data = raw_data.values[:,1:]
labels = raw_data.values[:,0]

get_ushapelet(data,6,10)
