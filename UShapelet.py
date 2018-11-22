import numpy as np
import pandas as pd
import pycuda.driver as dv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from sax_utils import cut_points


def get_ushapelet(data, splen, num_projections):
    """
    :param data: Time series data matrix in the shape (num_time_series,time_series_length)
    :param splen: Length of the shapelet to extract.
    :param num_projections: Number of random masking rounds to use.
    :return: Returns a new U-shapelet instance.
    """
    # Get subsequences of each time series
    # We can use indices here to ensure that we are not making unecessary copies of data.
    subseqs = get_subsequences(data,splen)

    # Get the SAX representations of each of the sub-sequences
    sax_subseqs = convert_to_SAX(subseqs)

    counts = np.zeros((sax_subseqs.shape[0],sax_subseqs.shape[1],num_projections))

    for i in range(num_projections):
        projections = get_random_projections(sax_subseqs)
        counts[:,:,i] = count_collisions(projections)

    results = filter_candidates(counts)
    print("Done")




def get_subsequences(data, splen):
    """
    :param data: A data matrix of shape (num_time_series, len_time_series).
    :param splen: Length of shapelets to extract.
    :return: The indices of the sub-subsequences in each time series.
    """


    # Unpack the data's shape tuple into two elements. The first dimension (# of rows) is the number of time series
    # objects. The second dimension is the length of each time series.
    ts_num,ts_len = data.shape

    # We slide the window across the entire time series, starting at index 0 and ending at index
    # ts_len-splen+1, which is the last index containing splen elements.
    subseq_idcs =  np.arange(0,ts_len-splen+1)
    subseq_ranges = np.array(list(map(lambda x : np.arange(x,x+splen),subseq_idcs)))
    subseqs = data[:,subseq_ranges]

    return subseqs

def convert_to_SAX(data, paa_len = 16, paa_vocab=4):
    """
    :param data: A Data matrix of (num_time_series, num_shapelet, length_shapelet).
    :return: The SAX representations of each shapelet in the matrix.
    """
    num_ts,num_sp,sp_len = data.shape

    # The PAA (Piecewise Aggregate Approximation) step will generate a PAA approximation of every sub-sequence.
    # This PAA representation will be a integer string of length paa_len, containing at most paa_vocab unique symbols.

    # Create an array to hold the means.
    means = np.zeros((num_ts,num_sp,paa_len))

    # Convert each sequence to a PAA approximation.
    if sp_len == paa_len:
        means[:] = data
    elif sp_len % paa_len == 0:
        pts_per_letter = sp_len // paa_len
        means[:] = np.mean(data.reshape(num_ts,num_sp,paa_len,pts_per_letter),axis=-1)
    else:
        tmp = data.repeat(paa_len).reshape(num_ts,num_sp,paa_len,sp_len)
        means[:] = np.mean(tmp,axis=-1)

    # TODO: remove the duplicate sequences here?
    # If there are any consecutive shapelets that have the same PAA approximation, we keep track of it so that we do
    # not hash the shapelet twice and count both.
    skip_idcs = np.where(means[:,:-1,:] == means[:,1:,:])
    means[skip_idcs] = -1

    # Convert the PAA representations to SAX discrete strings.
    sax_words = np.zeros_like(means,dtype=np.uint8)

    for ts in range(num_ts):
        for sp in range(num_sp):
            sax_words[ts,sp,:] = np.sum(np.array([cut_points[paa_vocab] <= x for x in means[ts,sp,:]]),axis=1)

    return sax_words



def get_random_projections(data,num_masked_elems = 3):
    """
    :param data: A Data matrix of SAX words in the shape (num_time_series, num_shapelet, paa_len).
    :param num_masked_elems: The number of elements to mask in one round of masking. Default is 3.
    :return: Random masked versions of the input shapelets
    """

    # Unpack the shape tuple.
    num_ts,num_sp,paa_len = data.shape

    # Choose random indices to mask. Sample without replacement so we don't choose the same index twice.
    mask_idcs = np.random.choice(paa_len,num_masked_elems,replace=False)

    # Make a copy of the data so that we can modify it by masking indices without changing the original.
    tmp = data.copy()
    tmp[:,:,mask_idcs] = 0

    # Return the masked SAX representation.
    return tmp


def count_collisions(projections):
    """
    :param projections: A data matrix of (num_time_series, num_shapelet, length_shapelet)
    :return: The collisions found for each random mask version of the shapelet
    """
    num_ts,num_sp,paa_len = projections.shape
    collisions = np.zeros((num_ts,num_sp))
    # Dictionary contains pairs of {SAX_Word -> first_index_appearance}
    collisions_map = {}
    for ts in range(num_ts):
        for sp in range(num_sp):
            tmp_string = "".join(map(str,projections[ts,sp,:]))
            if collisions_map.get(tmp_string) is None:
                collisions_map[tmp_string] = (ts,sp)
            else:
                collisions[collisions_map[tmp_string]] += 1
    print("Done!")
    return collisions

def filter_candidates(counts):
    """
    :param counts: Counts of the number of collisions for each shapelet
    :return: A sorted list of the best shapelets
    """
    num_ts,num_sp,num_masks = counts.shape
    lb = max(0.1*num_ts,2)
    ub = num_ts*0.9
    # Take the average of the number of collisions over the last axis of the array
    mean = np.mean(counts,axis=-1)

    cand_orig_idcs = np.argwhere(np.logical_and(mean > lb, mean < ub))
    cand_idcs = np.where(np.logical_and(mean > lb, mean < ub))

    cands = counts[cand_idcs]

    sorted_cand_indcs = np.argsort(np.std(cands,axis=1))

    idcs = cand_orig_idcs[sorted_cand_indcs].T

    return counts[idcs[0],idcs[1]]

def compute_gap(data):
    """
    :param data: A shapelet or set of shapelets
    :return: the gap score metric for this shapelet.
    """
    pass


