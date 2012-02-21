#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

#define TPB_X 16
#define TPB_Y 16
#define TPB (TPB_X * TPB_Y)

#define PPT_SQ_RT 2
#define PPT (PPT_SQ_RT * PPT_SQ_RT)

#define PPB_X (TPB_X * PPT_SQ_RT)
#define PPB_Y (TPB_Y * PPT_SQ_RT)
#define PPB (PPB_X * PPB_Y)

#define PREF_CIRC_LIST_SIZE 2500
#define CIRC_LIST_SIZE (2*TPB > PREF_CIRC_LIST_SIZE)?2*TPB:PREF_CIRC_LIST_SIZE

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    int numXRegions;
    int numYRegions;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void
shadePixel(int circleIndex, float2 pixelCenter, float3 p, float4* imagePtr) {

    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;

    float rad = cuConstRendererParams.radius[circleIndex];;
    float maxDist = rad * rad;

    // circle does not contribute to the image
    if (pixelDist > maxDist)
        return;

    float3 rgb;
    float alpha;

    // there is a non-zero contribution.  Now compute the shading value

    // suggestion: This conditional is in the inner loop.  Although it
    // will evaluate the same for all threads, there is overhead in
    // setting up the lane masks etc to implement the conditional.  It
    // would be wise to perform this logic outside of the loop next in
    // kernelRenderCircles.  (If feeling good about yourself, you
    // could use some specialized template magic).
    if (cuConstRendererParams.sceneName == SNOWFLAKES || cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

        const float kCircleMaxAlpha = .5f;
        const float falloffScale = 4.f;

        float normPixelDist = sqrt(pixelDist) / rad;
        rgb = lookupColor(normPixelDist);

        float maxAlpha = .6f + .4f * (1.f-p.z);
        maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
        alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    } else {
        // simple: each circle has an assigned color
        int index3 = 3 * circleIndex;
        rgb = *(float3*)&(cuConstRendererParams.color[index3]);
        alpha = .5f;
    }

    float oneMinusAlpha = 1.f - alpha;

    // BEGIN SHOULD-BE-ATOMIC REGION
    // global memory read

    float4 existingColor = *imagePtr;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;

    // global memory write
    *imagePtr = newColor;

    // END SHOULD-BE-ATOMIC REGION
}

__device__ __inline__ int
circleInBox(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // clamp circle center to box (finds the closest point on the box)
    float closestX = (circleX > boxL) ? ((circleX < boxR) ? circleX : boxR) : boxL;
    float closestY = (circleY > boxB) ? ((circleY < boxT) ? circleY : boxT) : boxB;

    // is circle radius less than the distance to the closest point on
    // the box?
    float distX = closestX - circleX;
    float distY = closestY - circleY;

    if ( ((distX*distX) + (distY*distY)) <= (circleRadius*circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}

__device__ __inline__ int
circleInBoxConservative(
    float circleX, float circleY, float circleRadius,
    float boxL, float boxR, float boxT, float boxB)
{

    // expand box by circle radius.  Test if circle center is in the
    // expanded box.

    if ( circleX >= (boxL - circleRadius) &&
         circleX <= (boxR + circleRadius) &&
         circleY >= (boxB - circleRadius) &&
         circleY <= (boxT + circleRadius) ) {
        return 1;
    } else {
        return 0;
    }
}

#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)
#define SCAN_BLOCK_DIM TPB

//Almost the same as naive scan1Inclusive, but doesn't need __syncthreads()
//assuming size <= WARP_SIZE
inline __device__ uint
warpScanInclusive(int threadIndex, uint idata, volatile uint *s_Data, uint size){
    uint pos = 2 * threadIndex - (threadIndex & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for(uint offset = 1; offset < size; offset <<= 1)
        s_Data[pos] += s_Data[pos - offset];

    return s_Data[pos];
}

inline __device__ uint warpScanExclusive(int threadIndex, uint idata, volatile uint *sScratch, uint size){
    return warpScanInclusive(threadIndex, idata, sScratch, size) - idata;
}

__inline__ __device__ void
sharedMemExclusiveScan(int threadIndex, uint* sInput, uint* sOutput, volatile uint* sScratch, uint size)
{
    if (size > WARP_SIZE) {

        uint idata = sInput[threadIndex];

        //Bottom-level inclusive warp scan
        uint warpResult = warpScanInclusive(threadIndex, idata, sScratch, WARP_SIZE);

        // Save top elements of each warp for exclusive warp scan sync
        // to wait for warp scans to complete (because s_Data is being
        // overwritten)
        __syncthreads();

        if ( (threadIndex & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
            sScratch[threadIndex >> LOG2_WARP_SIZE] = warpResult;

        // wait for warp scans to complete
        __syncthreads();

        if ( threadIndex < (SCAN_BLOCK_DIM / WARP_SIZE)) {
            // grab top warp elements
            uint val = sScratch[threadIndex];
            // calculate exclusive scan and write back to shared memory
            sScratch[threadIndex] = warpScanExclusive(threadIndex, val, sScratch, size >> LOG2_WARP_SIZE);
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();

        sOutput[threadIndex] = warpResult + sScratch[threadIndex >> LOG2_WARP_SIZE] - idata;

    } else if (threadIndex < WARP_SIZE) {
        uint idata = sInput[threadIndex];
        sOutput[threadIndex] = warpScanExclusive(threadIndex, idata, sScratch, size);
    }
}


// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

//    if(!(blockIdx.x == 16 && blockIdx.y == 16))
//        return;

    int threadIndex = threadIdx.y * TPB_X + threadIdx.x;

    __shared__ uint circle_list[CIRC_LIST_SIZE];
    __shared__ uint circle_list_count[TPB];
    __shared__ uint circle_list_index[TPB];
    
    float invWidth = 1.f / cuConstRendererParams.imageWidth;
    float invHeight = 1.f / cuConstRendererParams.imageHeight;

    //compute bounding box of region
    short region_xmin = PPB_X * blockIdx.x;
    short region_xmax = PPB_X * (blockIdx.x + 1) - 1;
    short region_ymin = PPB_Y * blockIdx.y;
    short region_ymax = PPB_Y * (blockIdx.y + 1) - 1;

    // convert to normalized float coords
    float boxL = invWidth * region_xmin;
    float boxR = invWidth * region_xmax;
    float boxB = invHeight * region_ymin;
    float boxT = invHeight * region_ymax;

    /*************************************************
     * Phase 1
     *   build circle-list for each region
     *************************************************/

    int numCircles = cuConstRendererParams.numCircles;

    uint private_circle_list[16];

    int privateCount = 0;
    for(int i = threadIndex; i <= numCircles; i += TPB) {
          int index3 = 3 * i;

          // read position and radius
          float3 p = *(float3*)(&cuConstRendererParams.position[index3]);
          float  rad = cuConstRendererParams.radius[i];

          if( circleInBox(p.x, p.y, rad, boxL, boxR, boxT, boxB) )
            private_circle_list[privateCount++] = i;
    }

    // store my result in circle_list_counts
    circle_list_count[threadIndex] = privateCount;
    __syncthreads();

    // perform prefix scan
    sharedMemExclusiveScan(threadIndex, circle_list_count, circle_list_index, circle_list, TPB);
    __syncthreads();

    // use my prefix scan index to store my circles
    int totalCount = circle_list_index[TPB-1] + circle_list_count[TPB-1];
    int myIndex = circle_list_index[threadIndex];

    for(int i = 0; i < privateCount; i++ ) {
        circle_list[myIndex++] = private_circle_list[i];
    }

    if(blockIdx.x == 1 && blockIdx.y == 1) {
        if(threadIndex == 0) {
            printf("total count: %u\n", totalCount);
            printf("privateCount:%u\n", privateCount);
        }
    } 

    /*************************************************
     * Phase 2
     *    render each pixel in this region based off circle-list
     *************************************************/
     
    __syncthreads();
    
    for(int c = 0; c < totalCount; c++) {
        int ci = circle_list[c];
        int index3 = 3 * ci;
        float3 pix = *(float3*)(&cuConstRendererParams.position[index3]);

    for(int pi=0; pi < PPT; pi++) {
        //calculate my pixel coordinates and ptrs
        int pixel_index = (threadIndex + TPB * pi); //interleaved
        int pixel_x = region_xmin + (pixel_index % PPB_X);
        int pixel_y = region_ymin + (pixel_index / PPB_X);
        float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixel_y * cuConstRendererParams.imageWidth + pixel_x)]);
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixel_x) + 0.5f), invHeight * (static_cast<float>(pixel_y) + 0.5f));

            shadePixel(ci, pixelCenterNorm, pix, imgPtr);
    }
    }

}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    numXRegions = 0;
    numYRegions = 0;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.
    numXRegions = ((image->width-1) / PPB_X) + 1;   // rounding up
    numYRegions = ((image->height-1) / PPB_Y) + 1; // rounding up

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.numXRegions = numXRegions;
    params.numYRegions = numYRegions;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaThreadSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {

        // 256 threads per block is a healthy number
        dim3 blockDim(256, 1);
        dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
        cudaThreadSynchronize();
    }
}

void
CudaRenderer::render() {
    // 256 threads per block is a healthy number
    dim3 blockDim(numXRegions, numYRegions);
    dim3 gridDim( TPB_X, TPB_Y );

    //printf("launching kernels %dx%d blks %dx%d threads/blk", blockDim.x, gridDim);
    printf("image size %d x %d\n", image->width, image->height);
    printf("launching kernels (%dx%d b @ %dx%d tpb)\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    printf("num circles: %d\n", numCircles);
    
    kernelRenderCircles<<<blockDim, gridDim>>>();
    cudaThreadSynchronize();
}
