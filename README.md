# ParallelFinalProject
A CUDA implementation of UShapelet-based Clustering. Completed as a project for CSCI-5551 (Parallel and Distributed Systems).

# Requirements
Requires Python 3 with the following dependencies:
* `numpy`
* `pandas`
* `scikit-learn` 
* `pycuda`
* `matplotlib`

# Usage
To use the program run `python3 Driver.py <input_file> <cuda_flag>` I have added two sample input files `Trace.txt` and `FourClasses.txt` to the repo.

# Implementation
I originally identified three areas for parallelization:
1. SAX Word Hashing
2. Hash Collision Checking
3. Gap Score Computation

I was able to parallelize two of my three original targets (SAX Word Hashing and Gap Score Computation). I was not able to complete the Hash Collision Checking parallelization in time.
However, my CUDA implementation runs much faster than the sequential version, even without the collision checking parallelization. This is because the 
sequential implementation spends the majority of its time performing Gap Score computations, which are much faster in the CUDA implementation.

# File Breakdown
1. `Driver.py`
    * This is the main file used to run the program.
2. `UShapelet.py` 
    * This file contains the implementations for UShapelet computation. I use a flag `use_cuda` to determine which sections of the code run in sequentially
and which parts run in parallel. Setting the `<cuda_flag>` parameter on the command line will toggle this on or off for all functions in the code.
3. `cuda_helper.py` 
    * This file wraps some of the boilerplate code for CUDA and PyCuda setup. The functions here allocate memory and copy parameters to the GPU before
    calling the CUDA kernels.
4. `cuda_modules.py`
    * This file contains the actual CUDA kernels used to perform each computation. There are two kernels for acclerating the PAA/SAX process and the Gap Score process,
    called `compute_PAA` and `compute_sdist` respectively.
    
# Results

## Output
This program uses a Rand Index score to compute the quality of the clustering generated by the algorithm. The Rand Index measures the similarity between two sets of clusterings. In our case, both
sample files come with cluster labels, so the Rand Index between the "true" labels and the algorithms labels is a good measure of the accuracy of our algorithm.
After each shapelet extracted, the program will print out the time taken (in seconds) to complete each section of the program.

For `FourClasses.txt` the algorithm clusters perfectly, with a Rand Index of 1. 
### Without CUDA Acceleration
<img src="https://github.com/Andrew0Hill/ParallelFinalProject/blob/master/nocuda/NoCudaOutput4Classes.PNG" width="30%" align="middle">

### With CUDA Acceleration
<img src="https://github.com/Andrew0Hill/ParallelFinalProject/blob/master/withcuda/CUDAOutput4Classes.PNG" width="30%" align="middle">

For `Trace.txt` the algorithm gets a Rand Index of ~0.71.
### Without CUDA Acceleration
<img src="https://github.com/Andrew0Hill/ParallelFinalProject/blob/master/nocuda/NoCudaOutput.PNG" width="30%" align="middle">

### With CUDA Acceleration
<img src="https://github.com/Andrew0Hill/ParallelFinalProject/blob/master/withcuda/CUDAOutput.PNG" width="30%" align="middle">

The following charts show the speedup gained from the parallelization. 
For this section, tests were run using the `Trace.txt` input file, as it is the larger of the two sample inputs. 
## Sequential Runtime per Section
<img src="https://github.com/Andrew0Hill/ParallelFinalProject/blob/master/nocuda/bar_chart.png" width="75%" align="middle">

## Parallel Runtime per Section

<img src="https://github.com/Andrew0Hill/ParallelFinalProject/blob/master/withcuda/bar_chart.png" width="75%" align="middle">

For `Trace.txt`, the sequential version takes almost 30 seconds to perform the Gap Score computation, while the CUDA version performs the same operation in less than one second.



