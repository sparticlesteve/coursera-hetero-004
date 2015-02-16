// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

//@@ insert code here

//
// Kernels
//

//-----------------------------------------------------------------------------
// Convert image array of float into uchar.
// TODO: consider assigning more than one element to threads, strided.
//-----------------------------------------------------------------------------
__global__ void convertToChar(float* input, unsigned char* output, int len)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < len)
        output[i] = (unsigned char) (255 * input[i]);
}

//-----------------------------------------------------------------------------
// Convert RGB image array into grayscale.
//-----------------------------------------------------------------------------
__global__ void convertToGrayScale(unsigned char* input, unsigned char* output,
                                   int numPixels)
{
    // Pixel index
    int i = (blockIdx*blockDim.x + threadIdx.x)*3;
    if(i < numPixels){
        // Combine red, blue, green to produce gray
        unsigned char r = input[i];
        unsigned char g = input[i+1];
        unsigned char b = input[i+2];
        // These factors were provided in the assignment
        output[i] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    }
}

//-----------------------------------------------------------------------------
// Main function
//-----------------------------------------------------------------------------
int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    unsigned char * deviceRGBData;
    const char * inputImageFile;

    //@@ Insert more code here

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    hostInputImageData = wbImage_getData(inputImage); // added
    hostOutputImageData = wbImage_getData(outputImage); // added
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    // Dump some information
    wbLog(INFO, "img width  ", imageWidth);
    wbLog(INFO, "img height ", imageHeight);
    wbLog(INFO, "img chans  ", imageChannels);
    
    // Allocate memory on device
    int numPixels = imageWidth*imageHeight;
    int imageLen = numPixels*imageChannels;
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck( cudaMalloc((void**)&deviceInputImageData, imageLen*sizeof(float)) );
    wbCheck( cudaMalloc((void**)&deviceOutputImageData, imageLen*sizeof(float)) );
    wbCheck( cudaMalloc((void**)&deviceRGBData, imageLen*sizeof(unsigned char)) );
    wbTime_stop(GPU, "Allocating GPU memory.");
    
    // Transfer input data to device
    wbTime_start(GPU, "Copying input to GPU.");
    wbCheck( cudaMemcpy(deviceInputImageData, hostInputImageData,
                        imageLen*sizeof(float), cudaMemcpyHostToDevice) );
    wbTime_stop(GPU, "Copying input to GPU.");
    
    //-------------------------------------------------------------------------
    // Begin kernel computations
    //-------------------------------------------------------------------------
    wbTime_start(Compute, "Performing kernel computations.");
    
    // Convert image data to unsigned char
    int blockSize1 = 1024;
    int gridSize1 = (imageLen-1) / blockSize1 + 1;
    wbLog(INFO, "Converting to uchar with blockSize", blockSize1, ", gridSize", gridSize1);
    convertToChar<<<gridSize1, blockSize1>>>(deviceInputImageData, deviceRGBData, imageLen);
    
    // Convert RGB to gray-scale
    int blockSize2 = 1024;
    int gridSize2 = (imageWidth*imageHeight-1) / blockSize2 + 1;
    wbLog(INFO, "Converting RGB to grayscale with blockSize", blockSize2, ", gridSize", gridSize2);
    convertToGrayScale(deviceRGBData, deviceGrayData, numPixels);

    // End kernel computations
    wbTime_stop(Compute, "Performing kernel computations.");

    // Free GPU memory
    wbTime_start(GPU, "Freeing GPU memory.");
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceRGBData);
    wbTime_stop(GPU, "Freeing GPU memory.");

    wbSolution(args, outputImage);

    //@@ insert code here

    return 0;
}
