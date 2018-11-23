import numpy as np
import pandas as pd
import pycuda.driver as dv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from utils import cut_points

ROOT_DIR = "C:/Users/96ahi/PycharmProjects/ParallelFinalProject/kk"

def convert_to_PAA(input, sp_len, paa_len):
    """
    :param input: The input array of data (num_time_series x time_series_len)
    :param sp_len: The length of each shapelet.
    :param paa_len: The length of the Piecewise Aggregate Approximation for each shapelet candidate
    :return: Each shapelet in Piecewise Aggregate Approximation form.
    """
    num_ts,len_ts = input.shape

    # Calculate the number of shapelets per time series
    sp_per_ts = len_ts - sp_len + 1
    # Create an array to hold the means.
    means = np.zeros((num_ts,sp_per_ts,paa_len))
    # Iterate through each time series.

    # In some cases, the PAA length is not divisible by shapelet length. To fix this problem,
    # we need to tile the shapelet values at each index in order to produce an output of size PAA length.
    # Example: paa_len = 5, sp_len = 3
    # Each shapelet is only three elements long, but we need to produce a PAA approximation with 5 elements.
    # The original Matlab code expands the data into a matrix of 3x5 by repeating each
    # element of the shapelet paa_len times.
    # paa_len times.
    #
    # Example Matrix:
    # For a length 3 shapelet with elements (d1, d2, d3)
    #
    #   d1 d1 d2 d2 d3
    #   d1 d1 d2 d3 d3
    #   d1 d2 d2 d3 d3
    #
    # Then we take the mean along the columns of the matrix, reducing it to a (1,5) output
    # which is what we wanted (A paa_len of 5).
    # This is not ideal for memory usage, however, because we must keep a matrix of (sp_len x paa_len)
    # in memory which could grow to be very large.
    for sp in range(sp_per_ts):
        shapelet = data[:,sp:sp+sp_len]
        means[:,sp] = np.mean(np.repeat(shapelet,paa_len,axis=1).reshape(num_ts,paa_len,sp_len).transpose(0,2,1),axis=1)
    return means

def convert_to_SAX(input,vcb_size=4):
    """
    :param input: Input data of (num_time_series x num_shapelets x shapelet_length)
    :param vcb_size: Vocab size. How many symbols to use in SAX word representation.
    Hardcoded as 4 in the UShapelet MATLAB implementation.
    :return: The entire string converted to SAX representation, with pointers to each sub-sequence
    """
    num_ts,num_shp,sp_len = input.shape
    sax_strings = []
    for ts in range(num_ts):
        sax_strings.append([])
        for sp in range(num_shp):
            sax_strings[-1].append(np.sum(np.array([cut_points[vcb_size] <= x for x in input[ts,sp,:]]),axis=1).astype(np.uint8))

    return sax_strings

def hash_all_shapelets(input):
    """
    :param input: list of lists of SAX words as input.
    :return:
    """
    ts_len = len(input)
    sp_num = len(input[0])
    sp_len = len(input[0][0])

    num_rounds = 10
    num_masks = 3

    dct = dict()

    for i in range(ts_len):
        for j in range(sp_num):
            for k in range(num_rounds):
                idcs = np.random.choice(sp_len,num_masks)
                temp = input[i][j].copy()
                temp[idcs] = 0
                temp_s = "".join(map(str, temp))
                if dct.get(temp_s) is None:
                    dct[temp_s] = 1
                else:
                    dct[temp_s] += 1
    return dct

def get_subsq_idcs(ts_len, splen):
    """
    :param ts_len: The length of a time-series in the dataset. Assumes that all time series are
                    the same length.
    :param splen: Length of subsequence to split into.
    :return: The indices of the subsequences for this data.
    """
    return np.arange(0,ts_len - splen + 2)

raw_data = pd.read_csv(ROOT_DIR + "vmu_v2_hourly_full_week2.csv",header=None)
data = raw_data.values[1:,1:].astype(np.float32)
means = np.mean(data,axis=1).reshape(-1,1)
std = np.std(data,axis=1).reshape(-1,1)
data = np.divide(np.subtract(data,means),std)
labels = raw_data.values[1:,0]


paa_words = convert_to_PAA(data,5,16)
sax_words = convert_to_SAX(paa_words)
dct = hash_all_shapelets(sax_words)


print("Done.")