import numpy as np
import time
cut_points = {
    2 : np.array([np.NINF, 0]),
    3 : np.array([np.NINF,-0.43,0.43]),
    4 : np.array([np.NINF,-0.67,0,0.67])
}

# Simple decorator to time each part of program execution
def runtime(f):
    def time_func(*args, **kwargs):
        st = time.clock()
        result = f(*args, **kwargs)
        rt = time.clock() - st
        return (rt,result)
    return time_func