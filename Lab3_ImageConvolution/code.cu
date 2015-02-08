#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define MASK_WIDTH  5
#define MASK_RADIUS (MASK_WIDTH/2)

#define INPUT_TILE_WIDTH 32
#define OUTPUT_TILE_WIDTH (INPUT_TILE_WIDTH - MASK_WIDTH + 1)

#define NUM_CHANNELS 3

//@@ INSERT CODE HERE

__global__ void tiledConvolutionKernel(float* dIn, float* dOut,
                                       int height, int width,
                                       const float* __restrict__ mask)
{

    // Output tile indices
    int yOut = blockIdx.y*OUTPUT_TILE_WIDTH + threadIdx.y;
    int xOut = blockIdx.x*OUTPUT_TILE_WIDTH + threadIdx.x;

    // Input tile indices for loading into shared storage are shifted
    int yIn = yOut - MASK_RADIUS;
    int xIn = xOut - MASK_RADIUS;

    // Shared input tile storage
    __shared__ float dShare[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH][NUM_CHANNELS];

    // Load the input tile into shared memory.
    for(int iCh = 0; iCh < NUM_CHANNELS; ++iCh){

        // Use boundary check for "ghost" tiles
        if(0 <= yIn && yIn < height &&
           0 <= xIn && xIn < width){
            dShare[threadIdx.y][threadIdx.x][iCh] =
                dIn[(yIn*width + xIn)*NUM_CHANNELS + iCh];
        }
        else{
            dShare[threadIdx.y][threadIdx.x][iCh] = 0.0f;
        }

    }

    // Synchronize threads before moving on
    __syncthreads();


    // Calculate output tile results
    if(threadIdx.y < OUTPUT_TILE_WIDTH && threadIdx.x < OUTPUT_TILE_WIDTH){

        // Loop over channels
        for(int k = 0; k < NUM_CHANNELS; ++k)
        {

            // Each thread will calculate only one result
            // by looping over elements in the mask.
            float output = 0.0f;
            for(int i = 0; i < MASK_WIDTH; ++i){
                for(int j = 0; j < MASK_WIDTH; ++j){
                    output += mask[i*MASK_WIDTH+j] * dShare[i+threadIdx.y][j+threadIdx.x][k];
                }
            }

            // Write this channel to global output if within boundary
            if(yOut < height && xOut < width)
                dOut[(yOut*width+xOut)*NUM_CHANNELS + k] = output;

        }

    }

}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbLog(INFO, "imageChannels ", imageChannels);
    wbLog(INFO, "imageWidth ", imageWidth);
    wbLog(INFO, "imageHeight ", imageHeight);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    wbCheck( cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)) );
    wbCheck( cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)) );
    wbCheck( cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float)) );
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    wbCheck( cudaMemcpy(deviceInputImageData, hostInputImageData,
                        imageWidth * imageHeight * imageChannels * sizeof(float),
                        cudaMemcpyHostToDevice) );
    wbCheck( cudaMemcpy(deviceMaskData, hostMaskData,
                        maskRows * maskColumns * sizeof(float),
                        cudaMemcpyHostToDevice) );
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");

    //@@ INSERT CODE HERE

    wbLog(INFO, "INPUT_TILE_WIDTH  ", INPUT_TILE_WIDTH);
    wbLog(INFO, "OUTPUT_TILE_WIDTH ", OUTPUT_TILE_WIDTH);

    // Block size is input tile size
    dim3 blockSize(INPUT_TILE_WIDTH, INPUT_TILE_WIDTH, 1);

    // Grid size must be calculated in terms of output tile size,
    // rather than block size.
    int gridWidth = (imageWidth-1) / (OUTPUT_TILE_WIDTH) + 1;
    int gridHeight = (imageHeight-1) / (OUTPUT_TILE_WIDTH) + 1;
    dim3 gridSize(gridWidth, gridHeight, 1);

    wbLog(INFO, "Launching (", gridWidth, "x", gridHeight,") blocks of size (",
          INPUT_TILE_WIDTH, "x", INPUT_TILE_WIDTH, ")");

    // Launch kernel
    tiledConvolutionKernel<<<gridSize, blockSize>>>(deviceInputImageData, deviceOutputImageData, 
                                                       imageHeight, imageWidth, deviceMaskData);

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    wbCheck( cudaMemcpy(hostOutputImageData, deviceOutputImageData,
                        imageWidth * imageHeight * imageChannels * sizeof(float),
                        cudaMemcpyDeviceToHost) );
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    // DEBUGGING
    //float* in = hostInputImageData;
    //float* out = hostOutputImageData;
    //float* m = hostMaskData;
    //int h = imageHeight;
    //int w = imageWidth;
    //int mw = MASK_WIDTH;

    /*wbLog(INFO, "Mask");
    for(int i=0; i<mw; ++i){
        int a = i*mw;
        wbLog(INFO, m[a], " ", m[a+1], " ", m[a+2], " ", m[a+3], " ", m[a+4]);
    }*/

    //wbLog(INFO, "results");
    //wbLog(INFO, in[0], " ", in[1], " ", in[2]);
    //wbLog(INFO, in[3], " ", in[4], " ", in[5]);
    //wbLog(INFO, in[6], " ", in[7], " ", in[8]);
    //wbLog(INFO, in[9], " ", in[10], " ", in[11]);
    //wbLog(INFO, in[12], " ", in[13], " ", in[14]);
    //wbLog(INFO, in[15], " ", in[16], " ", in[17]);

    /*for(int y = 0; y < 1; ++y){
        for(int x = 0; x < w/4; ++x){
            int i = (y*w + x)*3;
            wbLog(INFO, y, ",", x, "  rgbIn  ", in[i], " ", in[i+1], " ", in[i+2]);
            wbLog(INFO, y, ",", x, "  rgbO ", out[i], " ", out[i+1], " ", out[i+2]);
        }
    }*/

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
