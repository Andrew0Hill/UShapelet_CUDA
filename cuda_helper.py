import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
from cuda_modules import sdist_module
from math import ceil, floor

class CUDA_sdist(object):
    def __init__(self):
        self.threads_per_block = (8,32,1)
        self.cuda_sdist = sdist_module.get_function("compute_sdist")

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

        self.cuda_sdist(spl_gpu,data_gpu,dist_gpu,np.int32(sp_len),np.int32(ts_len),np.int32(ts_num),np.int32(sp_num), block=self.threads_per_block, grid=self.grid_size)

        cuda.memcpy_dtoh(distances, dist_gpu)

class CUDA_collisions(object):
    pass

