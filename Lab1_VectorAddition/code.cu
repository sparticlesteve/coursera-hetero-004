// MP 1
#include <wb.h>


// Error check
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)


__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
    //@@ Insert code to implement vector addition here
    // Calculate my thread ID, or index in the arrays
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < len){
        out[i] = in1[i] + in2[i];
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;
    float *deviceInput1;
    float *deviceInput2;
    float *deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = ( float * )malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");
  
    //@@ Allocate GPU memory here
    int size = inputLength * sizeof(float);
    wbCheck( cudaMalloc((void**)&deviceInput1, size) );
    wbCheck( cudaMalloc((void**)&deviceInput2, size) );
    wbCheck( cudaMalloc((void**)&deviceOutput, size) );

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    
    //@@ Copy memory to the GPU here
    wbCheck( cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice) );
    wbCheck( cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice) );

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    // 64 threads per block seems to have good
    // performance for the provided input datasets
    const int blockSize = 64;
    const int gridSize = (inputLength-1)/blockSize + 1;
    wbLog(INFO, "Block size: ", blockSize);
    wbLog(INFO, "Grid size:  ", gridSize);

    wbTime_start(Compute, "Performing CUDA computation");

    //@@ Launch the GPU Kernel here
    vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");

    //@@ Copy the GPU memory back to the CPU here
    wbCheck( cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost) );

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    
    //@@ Free the GPU memory here
    wbCheck( cudaFree(deviceInput1) );
    wbCheck( cudaFree(deviceInput2) );
    wbCheck( cudaFree(deviceOutput) );

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

