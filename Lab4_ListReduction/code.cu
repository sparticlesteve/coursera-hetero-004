// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void reductionKernel(float * input, float * output, int len)
{
    // This thread's index in the block
    unsigned int t = threadIdx.x;
    // Starting element of this block
    unsigned int start = 2*blockIdx.x*blockDim.x;
    
    //@@ Load a segment of the input vector into shared memory
    __shared__ float shared[2*BLOCK_SIZE];
    if(start + t < len)
        shared[t] = input[start+t];
    else
        shared[t] = 0.f;
    if(start + t + blockDim.x < len)
        shared[t+blockDim.x] = input[start+t+blockDim.x];
    else
        shared[t+blockDim.x] = 0.f;

    //@@ Traverse the reduction tree
    for(unsigned int s = blockDim.x; s > 0; s >>= 1){
        __syncthreads();
        // The first s elements accumulate respective values in the next s elements
        if(t < s) shared[t] += shared[t+s];
    }

    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    // Woops. All threads are doing this. Luckily there's no real race
    // condition. Just a waste of resources.
    // TODO: fix this!!
    output[blockIdx.x] = shared[0];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");

    //@@ Allocate GPU memory here
    int inputSize = numInputElements * sizeof(float);
    int outputSize = numOutputElements * sizeof(float);
    wbCheck( cudaMalloc((void**)&deviceInput, inputSize) );
    wbCheck( cudaMalloc((void**)&deviceOutput, outputSize) );

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");

    //@@ Copy memory to the GPU here
    wbCheck( cudaMemcpy(deviceInput, hostInput, inputSize, cudaMemcpyHostToDevice) );

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    int blockSize = BLOCK_SIZE;
    int gridSize = (numInputElements-1) / (blockSize*2) + 1;
    wbLog(INFO, "Block size: ", blockSize);
    wbLog(INFO, "Grid size:  ", gridSize);

    wbTime_start(Compute, "Performing CUDA computation");

    //@@ Launch the GPU Kernel here
    reductionKernel<<<gridSize, blockSize>>>
        (deviceInput, deviceOutput, numInputElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");

    //@@ Copy the GPU memory back to the CPU here
    wbCheck( cudaMemcpy(hostOutput, deviceOutput, outputSize,
                        cudaMemcpyDeviceToHost) );

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (int ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    wbTime_start(GPU, "Freeing GPU Memory");

    //@@ Free the GPU memory here
    wbCheck( cudaFree(deviceInput) );
    wbCheck( cudaFree(deviceOutput) );

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}


