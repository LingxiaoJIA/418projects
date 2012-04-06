#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "defines.h"

extern float toBW(int bytes, float sec);


__global__ void
chartest_kernel(float* distortions, int numDistortions, int maxDistortionSize, float* target, int tWidth, int tHeight, int numLocations, float* results) {
//    if(dID == 0)
//        printf("[T%d hello]\n", tid);

    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    
    int rangeW = tWidth - EDGE_DONT_BOTHER;

    int locId = blockId * NUM_THREADS_PER_BLOCK + threadId;
    if(locId >= numLocations)
        return;

    int tx_c = locId % rangeW;
    int ty_c = locId / rangeW;

//    int totalLocs = w_range * h_range;

    float maxVal = 0.0;
    for(int d = 0; d < numDistortions; d++) {
        // do all of the distortions at this location

        // calculate index of this distortion in buffer, pull out width/height
        int dIndex = (d * maxDistortionSize);
        int dWidth = int(distortions[dIndex++]);
        int dHeight = int(distortions[dIndex++]);
        
        // calculate location to work on
        int tx_0 = tx_c - (dWidth / 2);
        int ty_0 = ty_c - (dHeight / 2);
        if(tx_0 < 0 || ty_0 < 0 || tx_0 + dWidth >= tWidth || ty_0 + dHeight >= tHeight)
            continue;

        float sum_let = 0.0;
        float sum_conv = 0.0;
        for(int dy = 0; dy < dHeight; dy++) {
            for(int dx = 0; dx < dWidth; dx++) {
                /* calculate index into target buffer */
                int tx = tx_0 + dx;
                int ty = ty_0 + dy;
                int tIndex = ty * tWidth + tx;

                float t = target[tIndex];
                float d = distortions[dIndex++];

                /* Version 1
                 *   rewards matching as a percentage of pixels present */
                sum_let += d;
                sum_conv += (d * t);
                
                /* Version 2
                 *   rewards matching and punishes noise */
                //sum_let += d;
                //sum_conv += ( ((3 * d * t) - t) / 2.0);
                 
            }
        }
        float val = sum_conv / sum_let;
        if(val > maxVal) {
            maxVal = val;
        }
    }
    results[locId] = maxVal;

//    if(blockId == 0)
//        printf("[T%d found max of %f for (%d,%d) ]\n", threadId, maxVal, locX, locY);
    __syncthreads();

}

void
charTest(float * distortionsBuf, int numDistortions, int maxDistortionSize, float * targetBuf, int targetW, int targetH, int  numLocations, float * resultBuf) {

    const int targetBytes = targetW * targetH * sizeof(float);

    // compute number of blocks and threads per block
    const int threadsPerBlock = NUM_THREADS_PER_BLOCK;
    const int blocks = ((numLocations + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK);

    const int resultBytes = numLocations * sizeof(float);

    // allocate target buffer
    float* device_target;
    cudaMalloc(&device_target, targetBytes);

    // allocate letter buffers
    int distortionBytes = numDistortions * maxDistortionSize * sizeof(float);
    float * device_distortions;
    cudaMalloc( &device_distortions, distortionBytes );

    // allocate results buffer
    float * device_result;
    cudaMalloc( &device_result, resultBytes );

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    // copy target buffer
    cudaMemcpy(device_target, targetBuf, targetBytes, cudaMemcpyHostToDevice);
    // copy letter buffers
    cudaMemcpy(device_distortions, distortionsBuf, distortionBytes, cudaMemcpyHostToDevice);

    double kernelStartTime = CycleTimer::currentSeconds();
    // run kernel
    chartest_kernel<<<blocks, threadsPerBlock>>>(device_distortions, numDistortions, maxDistortionSize, device_target, targetW, targetH, numLocations, device_result);
    cudaThreadSynchronize();
    double kernelEndTime = CycleTimer::currentSeconds();

    // TODO copy result from GPU using cudaMemcpy
    cudaMemcpy( resultBuf, device_result, resultBytes, cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();
    
    double overallDuration = endTime - startTime;
    double kernelDuration = kernelEndTime - kernelStartTime;
    printf("\tOverall: %.3f ms\n", 1000.f * overallDuration);
    printf("\tKernel : %.3f ms\n", 1000.f * kernelDuration);
    
    // TODO free memory buffers on the GPU
    cudaFree(device_target);
    cudaFree(device_distortions);
    cudaFree(device_result);

}

void
printCudaInfo() {
    
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA devices\n", deviceCount);
    
    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
}
