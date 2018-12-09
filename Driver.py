import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import os
import sys
from UShapelet import get_ushapelet
from timekeeper import TimeKeeper
from sklearn.metrics import adjusted_rand_score
if __name__ == "__main__":

    ROOT_DIR = os.getcwd()
    try:
        _,data_file,cuda_select = sys.argv
    except Exception as e:
        print("Usage python3 Driver.py <filename> <use_cuda>")
        exit()
    # Run with CUDA if selected, otherwise don't.
    cuda_flag = True if cuda_select == "1" else False
    print("CUDA enable flag = %d" % cuda_flag)
    if data_file not in ["Trace.txt","FourClasses.txt"]:
        print("Running using custom file. File will be read as a CSV.")

    file_path = os.path.join(ROOT_DIR,data_file)

    #tk = TimeKeeper()

    # The 'Trace.txt' example requires a different parsing method because it is not a CSV file, but space delimited.
    if data_file == "Trace.txt":
        raw_data = pd.read_table(file_path,delim_whitespace=True,header=None)
        labels = raw_data.values[:,0].astype(np.int32)
    # Read the file in from CSV
    else:
        raw_data = pd.read_csv(file_path,header=None)
        # Split the data into a label set (the first column of the data),
        # and the actual data, (the second -> N columns).
        labels = raw_data.values[:,0]

    data = raw_data.values[:,1:]
    clust_labels = np.zeros(data.shape[0])
    clust_num = 1

    # In order to ensure that we extract only "good" clusters and do not need to assign every time series to a cluster,
    # we keep track of the gap score of the first shapelet we discover. At every iteration after the first, we check
    # that the gap score of the current shapelet is larger than 1/2 of the original gap score. If it is not, we stop
    # assigning clusters, and assume the rest of the time series are outliers which do not belong to any cluster.
    first_pass = True
    current_max_score = 0


    data_idcs = np.arange(data.shape[0])
    #temp_data = data
    while data_idcs.shape[0] > 0:
        # Extract a U-Shapelet from the data.
        idx,score,members,timings = get_ushapelet(data[data_idcs],30,10,tk=None,use_cuda=cuda_flag)

        if idx is None:
            print("Finished clustering. (No valid shapelets)")
            break
        elif score > current_max_score:
            current_max_score = score
        elif score < 0.5*current_max_score:
            print("Finished clustering. (Threshold epsilon)")
            break
        # Assign a cluster label to the members of the new cluster.
        clust_labels[data_idcs[members]] = clust_num
        # Increment the cluster number by one.
        clust_num += 1
        # Remove the members of this cluster from the data, and iterate again with the old data.
        data_idcs = np.delete(data_idcs,members,axis=0)
        print("Runtimes:")
        print("Subsequence: %f" % timings[0])
        print("SAX: %f" % timings[1])
        print("Collision Check: %f" % timings[2])
        print("Filter: %f" % timings[3])
        print("Gap Score: %f" % timings[4])
        print("Search: %f" % timings[5])
        print()
    print("%d clusters extracted." % np.unique(clust_labels).shape[0])
    print("Rand Index (Actual vs. Expected) %f" % adjusted_rand_score(labels,clust_labels))
    #tk.print_summary()
