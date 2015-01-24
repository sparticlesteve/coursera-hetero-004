#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    
    // Calculate my positions in the C matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if(row < numCRows && col < numCColumns){
        // My value at C[row][col] is given by the dot-product of A[row][] and B[][col]
        float tempC = 0;
        for(int i = 0; i < numAColumns; ++i){
            tempC += A[row*numAColumns + i] * B[col + i*numBColumns];
        }
    
        // Write back to global memory
        C[row*numCColumns + col] = tempC;
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;    // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA =
        ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB =
        ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    
    //@@ Allocate the hostC matrix
    hostC = new float[numCRows * numCColumns];
    
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    
    //@@ Allocate GPU memory here
    int sizeA = numARows * numAColumns * sizeof(float);
    int sizeB = numBRows * numBColumns * sizeof(float);
    int sizeC = numCRows * numCColumns * sizeof(float);
    wbCheck( cudaMalloc((void**)&deviceA, sizeA) );
    wbCheck( cudaMalloc((void**)&deviceB, sizeB) );
    wbCheck( cudaMalloc((void**)&deviceC, sizeC) );

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    
    //@@ Copy memory to the GPU here
    wbCheck( cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice) );
    wbCheck( cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice) );

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    
    // Choose a block size
    const int tileWidth = 16;
    dim3 blockSize(tileWidth, tileWidth, 1);
    // Choose grid size based on the block size and dataset
    dim3 gridSize((numCColumns-1)/tileWidth + 1, (numCRows-1)/tileWidth + 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    
    //@@ Launch the GPU Kernel here
    matrixMultiply<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC,
                                            numARows, numAColumns, 
                                            numBRows, numBColumns, 
                                            numCRows, numCColumns);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    
    //@@ Copy the GPU memory back to the CPU here
    wbCheck( cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost) );

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    
    //@@ Free the GPU memory here
    wbCheck( cudaFree(deviceA) );
    wbCheck( cudaFree(deviceB) );
    wbCheck( cudaFree(deviceC) );

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

