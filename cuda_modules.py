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
            sum = sqrt(sum)/sqrt((float)sp_len);
            min_sum = sum < min_sum ? sum : min_sum;
        } 
        distances[sp_idx * num_ts + ts_idx] = min_sum;
    }
  }
  
  /*
   *
   * BEGIN compute_PAA
   *
   */
  __global__ void compute_PAA(float* data, 
                              float* means,
                              float* cutoffs,
                              int* sax_words,
                              int num_ts, 
                              int num_sp, 
                              int ts_len, 
                              int sp_len, 
                              int paa_len,
                              int sax_vocab)
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
        
        // data_row_idx is the index into the time series array.
        int data_row_idx = data_x * ts_len;
        
        // means_row_idx is the index into the 3D array of means that we want to output.
        int means_row_idx = (data_y * paa_len) + (data_x * paa_len * num_sp);
        
        // In cases where the paa_len is not divisible by the sp_len (i.e. paa_len = 16 and sp_len = 5),
        // We must upsample (or downsample) the subsequence in order to fit it into a paa_len-size word. 
        // The authors of the paper do this by repeating the subsequence paa_len times, which creates a matrix with size
        // paa_len * sp_len, which is divisible by both paa_len and sp_len. I accomplish the same thing here without 
        // allocating any new memory by accumulating sums for each element of the PAA word, then taking the average.
        for(int i = 0; i < sp_len * paa_len; ++i){
            // Accumulate the values for this shapelet.
            means[(i/sp_len) + means_row_idx] += data[(i/paa_len) + data_row_idx + data_y];
            // Divide the accumulated sum by the shapelet length.
            if(i % sp_len == sp_len-1){
                means[(i/sp_len) + means_row_idx] = means[(i/sp_len) + means_row_idx]/sp_len;  
            }
        }
        
        // After means are computed, we need to map each PAA value to a discrete SAX symbol. We can do this in the same
        // kernel, and have each thread compute the SAX approximation for one shapelet.
        // The output sax_words array is of shape (num_ts, num_sp, paa_len). I reuse the means_row_idx here because
        // the means array and the sax_words array are the same shape.
        
        for(int i = 0; i < paa_len; i++){
            // Iterate through the cutoffs array, evaluate the statement, and sum it. 
            // A 'true' value will equal 1, so a value that is greater than the first cutoff will be assigned 1.
            // A value that is greater than the second cutoff will be assigned 1+1 = 2, and so on. 
            for(int j = 0; j < sax_vocab; ++j){
                sax_words[means_row_idx + i] += (means[means_row_idx + i] > cutoffs[j]);
            } 
        }
    }
  } 
  
  /*
   *
   * BEGIN count_collisions
   *
   */
  
""")

