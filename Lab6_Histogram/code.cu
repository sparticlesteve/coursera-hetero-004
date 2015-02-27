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


//-----------------------------------------------------------------------------
// Convert image array of float into uchar.
//-----------------------------------------------------------------------------
__global__ void convertToChar(float* input, unsigned char* output, int len)
{
    // Thread stride is total num of threads in the grid
    int stride = blockDim.x * gridDim.x;
    // Loop over my elements
    for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < len; i += stride)
        output[i] = (unsigned char) (255 * input[i]);
}

//-----------------------------------------------------------------------------
// Convert RGB image array into grayscale.
//-----------------------------------------------------------------------------
__global__ void convertToGrayScale(unsigned char* input, unsigned char* output,
                                   int numPixels)
{
    // Pixel index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPixels){
        // Combine red, blue, green to produce gray
        int inIdx = 3*i;
        unsigned char r = input[inIdx];
        unsigned char g = input[inIdx+1];
        unsigned char b = input[inIdx+2];
        // These factors were provided in the assignment
        output[i] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    }
}

//-----------------------------------------------------------------------------
// Calculate histogram from grayscale data
//-----------------------------------------------------------------------------
__global__ void computeHistogram(unsigned char* input, unsigned int* histo,
                                 int numPixels)
{
    // Shared copy of histogram for each thread block
    __shared__ unsigned int sharedHist[HISTOGRAM_LENGTH];
    // Initialize histogram
    if(threadIdx.x < HISTOGRAM_LENGTH) sharedHist[threadIdx.x] = 0;
    __syncthreads();
    // Thread stride is total num of threads in the grid
    int stride = blockDim.x * gridDim.x;
    // Loop over my elements
    for(int i = blockIdx.x*blockDim.x + threadIdx.x;
        i < numPixels; i += stride){
        atomicAdd(&sharedHist[input[i]], 1);
    }
    // Wait for the whole block to finish
    __syncthreads();
    // Accumulate results into output
    if(threadIdx.x < HISTOGRAM_LENGTH)
        atomicAdd(&histo[threadIdx.x], sharedHist[threadIdx.x]);
}

//-----------------------------------------------------------------------------
// Single-block scan kernel for calculating histogram CDF
//-----------------------------------------------------------------------------
__global__ void computeCDF(unsigned int* histo, float* cdf, int numPixels)
{
    // Thread index in the block
    unsigned int t = threadIdx.x;
    // Shared memory
    __shared__ unsigned int shared[HISTOGRAM_LENGTH];
    // Load two elements into shared memory
    if(t < HISTOGRAM_LENGTH)
        shared[t] = histo[t];
    else
        shared[t] = 0;
    if(t + blockDim.x < HISTOGRAM_LENGTH)
        shared[t+blockDim.x] = histo[t+blockDim.x];
    else
        shared[t+blockDim.x] = 0;
    __syncthreads();
    // Downsweep
    for(unsigned int s = 1; s <= blockDim.x; s *= 2){
        int index = (t+1)*s*2 - 1;
        if(index < 2*blockDim.x){
            shared[index] += shared[index-s];
        }
        __syncthreads();
    }
    // Upsweep
    for(int s = blockDim.x/2; s > 0; s /= 2){
        int index = (t+1)*s*2 - 1;
        if(index+s < 2*blockDim.x){
            shared[index+s] += shared[index];
        }
        __syncthreads();
    }
    // Write to global output
    // Scale with probability factor here
    float factor = 1. / numPixels;
    if(t < HISTOGRAM_LENGTH)
        cdf[t] = shared[t]*factor;
    if(t+blockDim.x < HISTOGRAM_LENGTH)
        cdf[t+blockDim.x] = shared[t+blockDim.x]*factor;
}

//-----------------------------------------------------------------------------
// Histogram equalization kernel uses the CDF to scale every pixel channel.
//-----------------------------------------------------------------------------
__global__ void equalizeImage(unsigned char* image, float* cdf, int len)
{
    // Thread stride is total num of threads in the grid
    int stride = blockDim.x * gridDim.x;
    float cdfMin = cdf[0];
    // Loop over my elements
    for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < len; i += stride){
        // get the uchar value at this element
        unsigned char val = image[i];
        // compute the corrected value
        int corVal = (int) ((cdf[val] - cdfMin)/(1. - cdfMin)*255.);
        corVal = min(corVal, 255);
        corVal = max(corVal, 0);
        image[i] = corVal;
    }
}

//-----------------------------------------------------------------------------
// Convert uchar image array back into float
//-----------------------------------------------------------------------------
__global__ void convertCharToFloat(unsigned char* input, float* output, int len)
{
    // Thread stride is total num of threads in the grid
    int stride = blockDim.x * gridDim.x;
    // Loop over my elements
    for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < len; i += stride)
        output[i] = (float) (input[i]/255.);
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
    const char * inputImageFile;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    unsigned char * deviceRGBData;
    unsigned char * deviceGrayData;
    unsigned int * deviceHist;
    float * deviceCDF;

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
    wbCheck( cudaMalloc((void**)&deviceGrayData, numPixels*sizeof(unsigned char)) );
    wbCheck( cudaMalloc((void**)&deviceHist, HISTOGRAM_LENGTH*sizeof(unsigned int)) );
    wbCheck( cudaMalloc((void**)&deviceCDF, HISTOGRAM_LENGTH*sizeof(float)) );
    wbTime_stop(GPU, "Allocating GPU memory.");

    // Initializing histogram
    wbCheck( cudaMemset(deviceHist, 0, HISTOGRAM_LENGTH*sizeof(unsigned int)) );

    // Transfer input data to device
    wbTime_start(GPU, "Copying input to GPU.");
    wbCheck( cudaMemcpy(deviceInputImageData, hostInputImageData,
                        imageLen*sizeof(float), cudaMemcpyHostToDevice) );
    wbTime_stop(GPU, "Copying input to GPU.");

    //-------------------------------------------------------------------------
    // Begin kernel computations
    //-------------------------------------------------------------------------
    wbTime_start(Compute, "Performing kernel computations.");

    // Debugging: dump out N input pixels
    /*const int nDump = 10;
    const int start = 0;
    float* h = hostInputImageData;
    printf("Input image data:");
    for(int i = start; i < start+nDump; ++i){
        int idx = 3*i;
        printf("  %i RGB %f, %f, %f\n", i, h[idx], h[idx+1], h[idx+2]);
    }*/

    // Convert image data to unsigned char. Strided kernel
    int blockSize1 = 512;
    int gridSize1 = ((imageLen-1) / blockSize1 + 1)/2;
    printf("Converting to uchar with %i blocks of %i threads\n",
           gridSize1, blockSize1);
    convertToChar<<<gridSize1, blockSize1>>>
        (deviceInputImageData, deviceRGBData,imageLen);

    // Debugging: dump the uchar
    /*unsigned char* hostRGBData = new unsigned char[imageLen];
    wbCheck( cudaMemcpy(hostRGBData, deviceRGBData,
                        imageLen*sizeof(unsigned char),
                        cudaMemcpyDeviceToHost) );
    printf("Input uchar RGB:\n");
    for(unsigned int i = start; i < start+nDump; ++i){
        int idx = 3*i;
        printf("  %i RGB %hhu, %hhu, %hhu\n", i,
               hostRGBData[idx], hostRGBData[idx+1], hostRGBData[idx+2]);
    }*/

    // Convert RGB to gray-scale
    int blockSize2 = 1024;
    int gridSize2 = (numPixels-1) / blockSize2 + 1;
    printf("Converting RGB to grayscale with %i blocks of %i threads\n",
           gridSize2, blockSize2);
    convertToGrayScale<<<gridSize2, blockSize2>>>
        (deviceRGBData, deviceGrayData, numPixels);

    // Debugging: dump the grayscale
    /*unsigned char * hostGrayData = new unsigned char[numPixels];
    wbCheck( cudaMemcpy(hostGrayData, deviceGrayData,
                        numPixels*sizeof(unsigned char),
                        cudaMemcpyDeviceToHost) );
    printf("Grayscale:\n");
    for(unsigned int i = start; i < start+nDump; ++i){
        printf("  %i Gray %hhu\n", i, hostGrayData[i]);
    }*/

    // Calculate the histogram. Strided kernel allows any block/grid size.
    int blockSize3 = 512;
    int gridSize3 = gridSize2/5; // each thread will do ~10 elements
    printf("Calculating histogram with %i blocks of %i threads\n",
           gridSize3, blockSize3);
    computeHistogram<<<gridSize3, blockSize3>>>
        (deviceGrayData, deviceHist, numPixels);

    // Debugging: copy histogram and dump
    /*unsigned int hostHist[HISTOGRAM_LENGTH];
    wbCheck( cudaMemcpy(hostHist, deviceHist,
                        HISTOGRAM_LENGTH*sizeof(unsigned int),
                        cudaMemcpyDeviceToHost) );
    printf("Histogram result:\n");
    for(unsigned int i = 0; i < HISTOGRAM_LENGTH; ++i){
        printf("  %i %i\n", i, hostHist[i]);
    }*/

    // Calculate the CDF via scan
    int blockSize4 = HISTOGRAM_LENGTH/2;
    int gridSize4 = 1;
    printf("Calculating CDF with %i blocks of %i threads\n",
           gridSize4, blockSize4);
    computeCDF<<<gridSize4, blockSize4>>>(deviceHist, deviceCDF, numPixels);

    // Debugging: copy CDF and dump
    /*float hostCDF[HISTOGRAM_LENGTH];
    wbCheck( cudaMemcpy(hostCDF, deviceCDF, HISTOGRAM_LENGTH*sizeof(float),
                        cudaMemcpyDeviceToHost) );
    printf("CDF result:\n");
    for(unsigned int i = 0; i < HISTOGRAM_LENGTH; ++i){
        printf("  %i %f\n", i, hostCDF[i]);
    }*/

    // Equalize the RGB image data using the CDF - strided kernel
    int blockSize5 = blockSize1;
    int gridSize5 = gridSize1;
    printf("Equalizing the RGB with %i blocks of %i threads\n",
           gridSize5, blockSize5);
    equalizeImage<<<gridSize5, blockSize5>>>
        (deviceRGBData, deviceCDF, imageLen);

    // Debugging: output uchar RGB data
    /*wbCheck( cudaMemcpy(hostRGBData, deviceRGBData,
                          imageLen*sizeof(unsigned char),
                          cudaMemcpyDeviceToHost) );
    printf("Output uchar RGB:\n");
    for(unsigned int i = start; i < start+nDump; ++i){
        int idx = 3*i;
        printf("  %i RGB %hhu, %hhu, %hhu\n", i,
               hostRGBData[idx], hostRGBData[idx+1], hostRGBData[idx+2]);
    }*/

    // Convert uchar image data back to float - strided kernel
    int blockSize6 = blockSize1;
    int gridSize6 = gridSize1;
    printf("Converting to float with %i blocks of %i threads\n",
           gridSize6, blockSize6);
    convertCharToFloat<<<gridSize6, blockSize6>>>
        (deviceRGBData, deviceOutputImageData, imageLen);

    wbCheck( cudaDeviceSynchronize() );

    // End kernel computations
    wbTime_stop(Compute, "Performing kernel computations.");

    // Copy output data back to host
    wbCheck( cudaMemcpy(hostOutputImageData, deviceOutputImageData,
                        imageLen*sizeof(float), cudaMemcpyDeviceToHost) );

    // Debugging: dump out N output pixels
    /*printf("Output image data:\n");
    h = hostOutputImageData;
    for(int i = start; i < start+nDump; ++i){
        int idx = 3*i;
        printf("  %i RGB %f, %f, %f\n", i, h[idx], h[idx+1], h[idx+2]);
    }*/

    // Check solution
    wbSolution(args, outputImage);

    // Free GPU memory
    wbTime_start(GPU, "Freeing GPU memory.");
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceRGBData);
    cudaFree(deviceGrayData);
    cudaFree(deviceHist);
    cudaFree(deviceCDF);
    wbTime_stop(GPU, "Freeing GPU memory.");

    // Free host memory
    wbImage_delete(inputImage);
    wbImage_delete(outputImage);

    // Free debug memory
    //delete[] hostRGBData;
    //delete[] hostGrayData;

    return 0;
}
