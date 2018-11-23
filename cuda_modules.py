from pycuda.compiler import SourceModule


sdist_module = SourceModule("""
  __global__ void compute_sdist(float *sbsq, float* ts, float* distances, int sp_len, int ts_len, int num_ts)
  {
    // Each thread receives one shapelet and one time series to compute the minimum distance for.
    int sp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ts_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Slide the shapelet over every window position in the time series.
    int num_wind = ts_len - sp_len + 1;
    float min_sum = 10000.0;
    float sum = 0;
    float dist = 0; 
    for(int i = 0; i < num_wind; ++i){
        sum = 0;
        for (int j = 0; j < sp_len; ++j){
            dist = sbsq[j] - ts[j+i]
            sum += dist * dist
        }
        min_sum = sum < min_sum ? sum : min_sum;
    } 
    distances[sp_idx + ts_idx * num_ts] = min_sum;
  }
""")