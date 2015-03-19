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

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    wbCheck( cudaMallocHost((void**)&hostOutput, inputLength*sizeof(float)) );
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    // Allocate memory on the device
    // Start with just one set of device arrays.
    // I should be able to do the sectioning on that directly
    int size = inputLength * sizeof(float);
    wbCheck( cudaMalloc((void**)&deviceInput1, size) );
    wbCheck( cudaMalloc((void**)&deviceInput2, size) );
    wbCheck( cudaMalloc((void**)&deviceOutput, size) );
        
    // Block and segment size
    // Segment size needs to be >= block size
    const int blockSize = 256;
    int segSize = 512;
    int gridSize = (segSize-1) / blockSize + 1;
    int numSegments = (inputLength-1) / segSize + 1;

    // Create the streams
    int nStreams = numSegments;
    cudaStream_t* streams = new cudaStream_t[nStreams];
    for(int i = 0; i < nStreams; ++i)
        wbCheck( cudaStreamCreate(&streams[i]) );
    
    printf("InputLength %i, nSeg %i, segSize %i, gridSize %i, blockSize %i\n",
           inputLength, numSegments, segSize, gridSize, blockSize);
    
    // Begin the loop over segments.
    // Each iteration of the loop will process one operation on each stream,
    // except at the boundaries. In order to ensure that the final streams
    // finish and are retrieved, we need to iterate 2 extra segments in the loop.
    for(int i = 0; i < numSegments + 2; ++i){

        printf("Loop iteration %i\n", i);

        // Schedule a return transfer.
        // Only performed starting from 3rd iteration.
        if(i >= 2){
            // My segment number: 2 streams previous
            int iSeg = i - 2;
            // My element offset
            int offset = iSeg*segSize;
            // Calculate the bytes to transfer
            int bytes = segSize*sizeof(float);
            if(offset + segSize > inputLength)
                bytes = (inputLength-offset)*sizeof(float);
            // Do the transfer
            printf("  retrieving segment %i, offset %i, bytes %i\n", iSeg, offset, bytes);
            wbCheck( cudaMemcpyAsync(hostOutput+offset, deviceOutput+offset, bytes, cudaMemcpyDeviceToHost, streams[iSeg]) );
        }
        
        // Schedule computation.
        if(i >= 1 && i < numSegments + 1){
            // My segment number: the previous stream
            int iSeg = i - 1;
            // My element offset
            int offset = iSeg*segSize;
            printf("  computing segment  %i, offset %i\n", iSeg, offset);
            vecAdd<<<gridSize, blockSize, 0, streams[iSeg]>>>(deviceInput1+offset, deviceInput2+offset, deviceOutput+offset, inputLength);    
        }
        
        // Scheduling outgoing transfer.
        // Not performed for last 2 iterations.
        if(i < numSegments){
            // My segment number: this one
            int iSeg = i;
            // My element offset
            int offset = iSeg*segSize;
            // Calculate the bytes to transfer
            int bytes = segSize*sizeof(float);
            if(offset + segSize > inputLength)
                bytes = (inputLength-offset)*sizeof(float);
            // Do the transfers
            printf("  sending segment    %i, offset %i, bytes %i\n", iSeg, offset, bytes);
            wbCheck( cudaMemcpyAsync(deviceInput1+offset, hostInput1+offset, bytes, cudaMemcpyHostToDevice, streams[iSeg]) );
            wbCheck( cudaMemcpyAsync(deviceInput2+offset, hostInput2+offset, bytes, cudaMemcpyHostToDevice, streams[iSeg]) );
        }
    
    }

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
    wbCheck( cudaFreeHost(hostOutput) );

    return 0;
}
