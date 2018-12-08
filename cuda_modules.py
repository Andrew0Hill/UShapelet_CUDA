from pycuda.compiler import SourceModule


cuda_module = SourceModule("""
  __global__ void compute_sdist(float *sbsq, 
                                float* ts, 
                                float* distances, 
                                int sp_len, 
                                int ts_len, 
                                int num_ts, 
                                int num_sp)
  {
    // Each thread receives one shapelet and one time series to compute the minimum distance for.
    int sp_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ts_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    
    if(sp_idx < num_sp && ts_idx < num_ts){
        // Slide the shapelet over every window position in the time series.
        int num_wind = ts_len - sp_len + 1;
        float min_sum = 3.4E38;
        float sum = 0;
        float dist = 0; 
        int sp_ind = sp_idx * sp_len;
        int ts_ind = ts_idx * ts_len;
        for(int i = 0; i < num_wind; ++i){
            sum = 0;
            for (int j = 0; j < sp_len; ++j){
                dist = sbsq[sp_ind + j] - ts[ts_ind + i + j];
                sum += dist * dist;
            }
            min_sum = sum < min_sum ? sum : min_sum;
        } 
        distances[sp_idx * num_ts + ts_idx] = min_sum;
    }
  }
  __global__ void compute_PAA(float* data, 
                              float* means,
                              float* cutoffs, 
                              int* sax, 
                              int num_ts, 
                              int num_sp, 
                              int ts_len, 
                              int sp_len, 
                              int paa_len, 
                              int sax_size)
  {
    // In this function we compute the PAA approximation of each shapelet. The input data
    // is the array of shapelets. We compute the PAA representation for each and return the result
    // in 'means', which can be reshaped to (num_ts,num_sp,sp_len) on the Python side.
    
    
    // Again we will use a 2D block, where the X dimension handles a time series,
    // and the Y dimension handles a shapelet within that time series.    
    int data_x = blockIdx.x * blockDim.x + threadIdx.x;
    int data_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(data_x < num_ts && data_y < num_sp){
        // Output 'mean' array should be of shape (num_ts, num_sp, sp_len)
        // Width = num_sp
        // Depth = sp_len
        //
        // Formula for flat 3D indexing is:
        //
        //      arr[x,y,z] = z + y*Depth + x*Depth*Width
        //
        
        // This is the base index of the shapelet in the data array.
        // We iterate through a shapelet by starting at base_idx and iterating until
        // base_idx + sp_len.
        int data_row_idx = data_x * ts_len;
        int means_row_idx = (data_y * paa_len) + (data_x * paa_len * num_sp);
        
        for(int i = 0; i < sp_len * paa_len; ++i){
            // Accumulate the values for this shapelet.
            means[(i/sp_len) + means_row_idx] += data[(i/paa_len) + data_row_idx];
            // Divide the accumulated sum by the shapelet length.
            if(i % sp_len == sp_len-1){
                means[(i/sp_len) + means_row_idx] = means[(i/sp_len) + means_row_idx]/sp_len;  
                for(int k = 0; k < sax_size; ++k){
                    sax[(i/sp_len) + means_row_idx] += (means[(i/sp_len) + means_row_idx] > cutoffs[k]);
                }
            }
        }
    }
  } 
""")

