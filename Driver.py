import pandas as pd
import numpy as np
import os
from UShapelet import get_ushapelet
from timekeeper import TimeKeeper

FILE_NAME = "FourClasses.txt"
ROOT_DIR = "/tmp/pycharm_project_926"
file_path = os.path.join(ROOT_DIR,FILE_NAME)

tk = TimeKeeper()

# Read the file in from CSV
raw_data = pd.read_csv(file_path,header=None)

# Split the data into a label set (the first column of the data),
# and the actual data, (the second -> N columns).
data = raw_data.values[:,1:]
labels = raw_data.values[:,0]

clust_labels = np.zeros(data.shape[0])
clust_num = 1

# In order to ensure that we extract only "good" clusters and do not need to assign every time series to a cluster,
# we keep track of the gap score of the first shapelet we discover. At every iteration after the first, we check
# that the gap score of the current shapelet is larger than 1/2 of the original gap score. If it is not, we stop
# assigning clusters, and assume the rest of the time series are outliers which do not belong to any cluster.
first_pass = True
orig_score = 0

temp_data = data
while temp_data.shape[0] > 0:
    # Extract a U-Shapelet from the data.
    idx,score,members = get_ushapelet(temp_data,5,10, tk=tk)

    if idx is None:
        print("Finished clustering.")
        break
    # Check if this is the first iteration or not.
    if first_pass:
        orig_score = score
        first_pass = False
    else:
        if score < 0.5*orig_score:
            print("Finished clustering.")
            break
    # Assign a cluster label to the members of the new cluster.
    clust_labels[members] = clust_num
    # Increment the cluster number by one.
    clust_num += 1
    # Remove the members of this cluster from the data, and iterate again with the old data.
    temp_data = np.delete(temp_data,members,axis=0)

print("%d clusters extracted." % (np.unique(clust_labels).shape[0] - 1))
tk.print_summary()
