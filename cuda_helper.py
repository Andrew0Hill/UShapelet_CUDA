import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
from cuda_modules import cuda_module
from math import ceil, floor

class CUDA_sdist(object):
    def __init__(self):
        self.threads_per_block = (8,32,1)
        self.cuda_sdist = cuda_module.get_function("compute_sdist")

    def __call__(self, shapelets, data, distances, sp_len, ts_len, ts_num, sp_num, grid_size = None):

        # Check if grid size was specified or not
        if grid_size:
            self.grid_size = grid_size
        else:
            self.grid_size = (ceil(sp_num/self.threads_per_block[0]), ceil(ts_num/self.threads_per_block[1]))

        # Allocate memory on the GPU for the calculation
        spl_gpu = cuda.mem_alloc(shapelets.nbytes)
        data_gpu = cuda.mem_alloc(data.nbytes)
        dist_gpu = cuda.mem_alloc(distances.nbytes)

        cuda.memcpy_htod(spl_gpu, shapelets)
        cuda.memcpy_htod(data_gpu, data)

        self.cuda_sdist(spl_gpu,
                        data_gpu,
                        dist_gpu,
                        np.int32(sp_len),
                        np.int32(ts_len),
                        np.int32(ts_num),
                        np.int32(sp_num),
                        block=self.threads_per_block,
                        grid=self.grid_size)

        cuda.memcpy_dtoh(distances, dist_gpu)

class CUDA_SAX_hash(object):
    def __init__(self):
        self.threads_per_block = (8,8,1)
        self.cuda_sax_hash = cuda_module.get_function("compute_PAA")

    def __call__(self, data, means, cutoffs, sax_words, num_ts, num_sp, ts_len, sp_len, paa_len, sax_size, grid_size = None):

        if grid_size:
            self.grid_size = grid_size
        else:
            self.grid_size = (ceil(num_ts/self.threads_per_block[0]),ceil(num_sp/self.threads_per_block[1]))

        # Allocate space for the data, which is a (num_ts,len_ts) array of time series.
        data_gpu = cuda.mem_alloc(data.nbytes)

        cutoffs_gpu = cuda.mem_alloc(cutoffs.nbytes)

        sax_words_gpu = cuda.mem_alloc(sax_words.nbytes)

        # Allocate space for the means array, which is (num_ts,len_ts,paa_len) in size.
        means_gpu = cuda.mem_alloc(means.nbytes)

        # Only 'data' and 'cutoffs' are inputs, so these are the only two we copy over to the GPU.
        cuda.memcpy_htod(data_gpu,data)
        cuda.memcpy_htod(cutoffs_gpu,cutoffs)
        # Call the function on the GPU.
        self.cuda_sax_hash(data_gpu,
                           means_gpu,
                           cutoffs_gpu,
                           sax_words_gpu,
                           np.int32(num_ts),
                           np.int32(num_sp),
                           np.int32(ts_len),
                           np.int32(sp_len),
                           np.int32(paa_len),
                           np.int32(sax_size),
                           block=self.threads_per_block,
                           grid=self.grid_size)

        cuda.memcpy_dtoh(means,means_gpu)
        cuda.memcpy_dtoh(sax_words,sax_words_gpu)