import numpy as np
from math import ceil,floor
import pandas as pd
import pycuda.driver as dv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from sax_utils import cut_points
from matplotlib import pyplot as plt

cluster_num = 0
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

    # Returns a sorted list of indices
    result_idcs = filter_candidates(counts)


    # Get the actual shapelets that are referenced by the indices we got from the candidate filtering.
    shapelet_cands = subseqs[result_idcs[0],result_idcs[1],:]

    # Compute the gap score for this set of shapelets.
    scores,distances,dt = compute_gap(shapelet_cands,data)

    # Get the indices of the members of this cluster in the original dataset.
    shapelet_idx,cluster_members = find_best_shapelet(scores,distances,dt)

    ts_idx,sp_idx = result_idcs[:,shapelet_idx]
    shapelet_subseq = subseqs[ts_idx,sp_idx,:]
    # Draw a figure of the shapelet that we extracted.
    global cluster_num
    plt.figure(figsize=(9,7))
    plt.title("Extracted Shapelet")
    plt.plot(data[ts_idx])
    plt.plot(range(sp_idx,sp_idx+splen),shapelet_subseq)
    plt.legend(["Parent TS", "Extracted Shapelet"])
    plt.savefig("Cluster_%d.png" % cluster_num)
    cluster_num += 1
    # Return these indices along with the shapelet that we found.
    return cluster_members
    print("Done")


def find_best_shapelet(scores,distances,dt):

    # Find the maximum gap score from our shapelet candidates.
    best_gap_idx = np.argmax(scores)
    # Find the distance cutoff associated with this gap score.
    dist_cutoff = dt[best_gap_idx]
    # Find the indices of the time series that belong to this cluster.
    # i.e. Find the indices in the distances array where the distance between the TS and the shapelet is below the
    # threshold
    member_idcs = np.argwhere(distances[best_gap_idx] <= dist_cutoff)

    # Return the index of the shapelet with the best gap score, and the indices of the time series which belong to
    # this cluster.
    return best_gap_idx,member_idcs[:,0]


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

    return idcs

def compute_gap(shapelets,data):
    """
    :param shapelets: A set of shapelets to compute the gap score for.
    :param data: The data matrix of all time series data. Should be of shape (num_time_series, length_time_series)
    :return: the gap score metric for this shapelet or shapelets.
    """

    lb = 2

    num_sp,len_sp = shapelets.shape

    num_ts,len_ts = data.shape

    # We will calculate the subsequence distance (sDist() in the paper) to all time series in the dataset.
    # sDist is defined as the minimum distance between a subsequence and a time series over all positions of that
    # subsequence in the time series. We will keep track of the sDist to every time series for every shapelet.
    distances = np.zeros((num_sp, num_ts))

    # First compute the subsequence distance from this shapelet to every sequence of this length in the dataset.
    # Iterate through all shapelets
    for i in range(num_sp):
        # Iterate through all possible time series
        for j in range(num_ts):
            min_dist = np.inf
            # Iterate through all possible positions of this shapelet
            # There are data.shape[1] - sp_len + 1 of these positions.
            for k in range(len_ts - len_sp + 1):
                dist = np.sum(np.square(shapelets[i] - data[j,k:k+len_sp]))
                if dist < min_dist:
                    min_dist = dist
            distances[i,j] = min_dist

    sorted_dists = np.sort(distances,axis=1)
    # No, I don't know why
    startPoint = ceil(sorted_dists.shape[1]*0.167)-1
    endPoint = floor(sorted_dists.shape[1]*0.833)-1

    # Array to hold the gap scores for each shapelet
    scores = np.zeros(num_sp)
    # Array to hold the dt, or distance cutoff that separates group A from group B.
    dt = np.zeros(num_sp)
    for i in range(num_sp):
        for j in range(startPoint,endPoint):
            d = sorted_dists[i,j]
            A_idcs = d >= distances[i]
            Da = distances[i,A_idcs]
            Db = distances[i,~A_idcs]
            r = Da.shape[0]/Db.shape[0]

            if ((0.2 < r) and (r < 5)):
                ma = np.mean(Da)
                mb = np.mean(Db)
                sa = np.std(Da)
                sb = np.std(Db)
                gap = mb - sb - (ma + sa)

                if gap > scores[i]:
                    scores[i] = gap
                    dt[i] = d

    return scores,distances,dt



