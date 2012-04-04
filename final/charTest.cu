#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "defines.h"

extern float toBW(int bytes, float sec);


__global__ void
chartest_kernel(float* distortions, float* target, int targetW, int targetH, int numLocations, float* results) {
//    if(dID == 0)
//        printf("[T%d hello]\n", tid);

    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    
    int rangeW = targetW - LETTER_WIDTH;
    //int rangeH = targetH - LETTER_HEIGHT;

    int locId = blockId * NUM_THREADS_PER_BLOCK + threadId;
    if(locId >= numLocations)
        return;

    int locX = locId % rangeW;
    int locY = locId / rangeW;

//    int totalLocs = w_range * h_range;

    float maxVal = 0.0;
    for(int d = 0; d < NUM_DISTORTIONS; d++) {
        // do all of the distortions at this location

        int x_0 = locX;
        int y_0 = locY;

        float sum_let = 0.0;
        float sum_conv = 0.0;
        for(int y = 0; y < LETTER_HEIGHT; y++) {
            for(int x = 0; x < LETTER_WIDTH; x++) {
                /* calculate index into letter buffer */
                int dIdx = distortionsIndex(d, x, y);

                /* calculate index into target buffer */
                int t_x = x_0 + x;
                int t_y = y_0 + y;
                int tIdx = t_y * targetW + t_x;

                sum_let += distortions[dIdx];
                sum_conv += (distortions[dIdx] * target[tIdx]);
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
charTest(float * distortionsBuf, float * targetBuf, int targetW, int targetH, int  numLocations, float * resultBuf) {

    const int targetBytes = targetW * targetH * sizeof(float);

    // compute number of blocks and threads per block
    const int threadsPerBlock = NUM_THREADS_PER_BLOCK;
    const int blocks = ((numLocations + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK);

    const int resultBytes = numLocations * sizeof(float);

    // allocate target buffer
    float* device_target;
    cudaMalloc(&device_target, targetBytes);

    // allocate letter buffers
    float * device_distortions;
    cudaMalloc( &device_distortions, DISTORTIONS_BUFFER_BYTES );

    // allocate results buffer
    float * device_result;
    cudaMalloc( &device_result, resultBytes );

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    // copy target buffer
    cudaMemcpy(device_target, targetBuf, targetBytes, cudaMemcpyHostToDevice);
    // copy letter buffers
    cudaMemcpy(device_distortions, distortionsBuf, DISTORTIONS_BUFFER_BYTES, cudaMemcpyHostToDevice);

    double kernelStartTime = CycleTimer::currentSeconds();
    // run kernel
    chartest_kernel<<<blocks, threadsPerBlock>>>(device_distortions, device_target, targetW, targetH, numLocations, device_result);
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
