// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 256

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)


// This kernel does a single-block inclusive scan, writing the result
// to the corresponding section of the output array. It uses shared memory.
__device__ void scanKernel(float * input, float * output, int len)
{
    // Thread index in the block
    unsigned int t = threadIdx.x;
    // Starting element of this block
    unsigned int start = 2*blockIdx.x*blockDim.x;
    
    // Define the shared memory (now dynamic!)
    extern __shared__ float shared[];

    // Load one element from first half of array.
    // Use zero if outside of array range.
    if(start+t < len)
        shared[t] = input[start+t];
    else 
        shared[t] = 0.f;

    // Load one element from second half of array.
    // Use zero if outside of array range.
    if(start + t + blockDim.x < len)
        shared[t+blockDim.x] = input[start+t+blockDim.x];
    else
        shared[t+blockDim.x] = 0.f;
    
    // Synchronize all threads before starting computation
    __syncthreads();
    
    // Downsweep phase, or reduction phase
    // As with reduction, the stride starts at 1 and doubles
    // until it's half the number of elements
    for(int s = 1; s <= blockDim.x; s *= 2){
        // Map consecutive threads onto desired elements
        int index = (t+1)*s*2 - 1;
        // Make sure the index is within the range
        if(index < 2*blockDim.x){
            // Accumulate element at -s away
            shared[index] += shared[index-s];
        }
        // Make sure all threads are done before next iteration
        __syncthreads();
    }
    
    // Upsweep phase, or post-reduction reverse phase
    // Now, stride starts at 1/4 the input size and halves until 1
    for(int s = blockDim.x/2; s > 0; s /= 2){
        // Index is same as downsweep, though we will actually be
        // accumulating in the other element instead.
        int index = (t+1)*s*2 - 1;
        // Make sure the other element is within range
        if(index+s < 2*blockDim.x){
            // Accumulate this index into element at +s away
            shared[index+s] += shared[index];
        }
        __syncthreads();
    }

    // Finally, write elements to global output.
    if(start+t < len)
        output[start+t] = shared[t];
    if(start+t+blockDim.x < len)
        output[start+t+blockDim.x] = shared[t+blockDim.x];
}


// This is a wrapper kernel for the first step of the multi-block scan,
// which calls the above scan kernel and writes the sum result to one
// element of the auxiliary array, indexed by blockIdx
__global__ void scanAndFillAux(float* input, float* output, float* aux, int len)
{
    // Execute the single-block scan kernel
    scanKernel(input, output, len);
    // Use one thread to write the block's sum to the aux array
    if(threadIdx.x == 0){
        // The block sum is in the final element of the block's array.
        unsigned int end = 2*blockDim.x*(blockIdx.x+1) - 1;
        // Write to aux array
        aux[blockIdx.x] = output[end];
    }
}


// This is a wrapper kernel which simply calls the single-block
// scan kernel; nothing more
__global__ void scan(float * input, float * output, int len)
{
    // For testing, need to duplicate the scan kernel for the aux scan
    scanKernel(input, output, len);
}


// Simple kernel for adding the aux array results to the full array.
// Each thread writes two elements, similar to the shared memory
// loading done in the scan kernel above.
__global__ void addAuxToFinal(float* output, float* aux, int len)
{
    // Thread and block indices
    unsigned int t = threadIdx.x;
    unsigned int b = blockIdx.x;
    // Skip the entire first block (kinda silly, I know)
    if(b > 0){
        // Starting element of this block
        unsigned int start = 2*b*blockDim.x;
        // Share the aux value for all threads in a block
        __shared__ float auxVal;
        // First thread loads the aux value of the previous block
        if(t == 0) auxVal = aux[b-1];
        __syncthreads();
        //float auxVal = aux[b-1];
        // Update the first value
        if(start+t < len)
            output[start+t] += auxVal;
        // Update the second value
        if(start+t+blockDim.x < len)
            output[start+t+blockDim.x] += auxVal;
    }
}


// Utility function for computing the next smallest power of two
int nextPowerOfTwo(int x)
{
    int y;
    for(y = 1; y < x; y <<= 1){}
    return y;
}


// Main function
int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    //float * hostAux; // for debugging
    float * deviceAux;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements is ", numElements);
    
    // Initialize block and grid size here; necessary for allocating the aux array.
    int blockSize = BLOCK_SIZE;
    int gridSize = (numElements-1) / (blockSize*2) + 1;
    int numAuxElements = nextPowerOfTwo(gridSize);
    int auxBlockSize = numAuxElements/2;
    wbLog(INFO, "Block size: ", blockSize);
    wbLog(INFO, "Grid size:  ", gridSize);
    wbLog(INFO, "Aux size:   ", numAuxElements);
    
    // Allocate host aux array for debugging
    //hostAux = (float*) malloc(numAuxElements * sizeof(float));

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceAux, numAuxElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbCheck(cudaMemset(deviceAux, 0, numAuxElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");

    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    
    // Scan each block and fill the aux array
    scanAndFillAux<<<gridSize, blockSize, blockSize*2*sizeof(float)>>>
        (deviceInput, deviceOutput, deviceAux, numElements);
    cudaDeviceSynchronize();

    // Run scan on the aux array
    // This will only work if the aux problem fits into one block.
    // Using one aux array and scanning in-place.
    scan<<<1, auxBlockSize, numAuxElements*sizeof(float)>>>(deviceAux, deviceAux, numAuxElements);
    cudaDeviceSynchronize();

    // Add the aux array elements into the corresponding block sections
    addAuxToFinal<<<gridSize, blockSize>>>(deviceOutput, deviceAux, numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    //wbCheck(cudaMemcpy(hostAux, deviceAux, numAuxElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");
    
    // Debugging the aux array
    /*wbLog(INFO, "Aux array elements:");
    for(int i=0; i<numAuxElements; ++i){
        wbLog(INFO, " ", i, " ", hostAux[i]);
    }*/

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceAux);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);
    //free(hostAux);

    return 0;
}
