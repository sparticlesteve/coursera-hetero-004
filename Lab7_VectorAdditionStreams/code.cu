#include    <wb.h>

// Error check
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

// Vector addition kernel
__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < len)
        out[i] = in1[i] + in2[i];
}

// Function for transferring a chunk asyncronously via a stream
//cudaError_t copySegment(float* out, float* in, int n, cudaMemcpyKind kind, cudaStream_t stream)


int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    // TODO: try pinned host memory
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    // Allocate memory on the device
    // Start with just one set of device arrays.
    // I should be able to do the sectioning on that directly
    int size = inputLength * sizeof(float);
    wbCheck( cudaMalloc((void**)&deviceInput1, size) );
    wbCheck( cudaMalloc((void**)&deviceInput2, size) );
    wbCheck( cudaMalloc((void**)&deviceOutput, size) );
    
    // Create the streams.
    // Let's now do 2 streams.
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    
    // Block and segment size
    // Start with one segment
    const int blockSize = 256;
    int segSize = inputLength/2;
    int gridSize = (segSize-1) / blockSize + 1;
    
    printf("InputLength %i, blockSize %i, segSize %i, gridSize %i\n",
           inputLength, blockSize, segSize, gridSize);

    // Asynchronous transfer of stream0
    wbCheck( cudaMemcpyAsync(deviceInput1, hostInput1, segSize*sizeof(float), cudaMemcpyHostToDevice, stream0) );
    wbCheck( cudaMemcpyAsync(deviceInput2, hostInput2, segSize*sizeof(float), cudaMemcpyHostToDevice, stream0) );
    
    // Asynchronous transfer of stream1
    wbCheck( cudaMemcpyAsync(deviceInput1+segSize, hostInput1+segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, stream1) );
    wbCheck( cudaMemcpyAsync(deviceInput2+segSize, hostInput2+segSize, segSize*sizeof(float), cudaMemcpyHostToDevice, stream1) );

    // Perform computation
    vecAdd<<<gridSize, blockSize, 0, stream0>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    vecAdd<<<gridSize, blockSize, 0, stream1>>>(deviceInput1+segSize, deviceInput2+segSize, deviceOutput+segSize, inputLength);
    
    // Asynchronous return transfer
    wbCheck( cudaMemcpyAsync(hostOutput, deviceOutput, segSize*sizeof(float), cudaMemcpyDeviceToHost, stream0) );
    wbCheck( cudaMemcpyAsync(hostOutput+segSize, deviceOutput+segSize, segSize*sizeof(float), cudaMemcpyDeviceToHost, stream1) );

    // Wait for remaining streams to finish
    wbCheck( cudaDeviceSynchronize() );
    
    // Debugging result
    const int nDump = 5;
    printf("Results\n");
    for(int i = 0; i < nDump; ++i){
        printf("  %i %f %f %f\n", i, hostInput1[i], hostInput2[i], hostOutput[i]);
    }

    // Check solution
    wbSolution(args, hostOutput, inputLength);
    
    // Free device memory
    wbCheck( cudaFree(deviceInput1) );
    wbCheck( cudaFree(deviceInput2) );
    wbCheck( cudaFree(deviceOutput) );

    // Free host memory
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
