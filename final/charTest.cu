#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "defines.h"

extern float toBW(int bytes, float sec);

void
chartest_kernel_sequential(float* distortions, int numDistortions, int maxDistortionSize, float* target, int tWidth, int tHeight, int numLocations, float* results, int blockId, int threadId) {
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
#ifdef v1
                sum_let += d;
                sum_conv += (d * t);
#endif
                
                /* Version 2
                 *   rewards matching, punishes noise and missing */
#ifdef v2
                sum_let += d;
                float match = d*t;
                float noise = (1-d)*t;
                float missing = d*(1-t);
                sum_conv += (match - 0.3*noise - 0.7*missing);
#endif

                /* Version 3
                 *   closeness of nearest pixel */
#ifdef v3
                if(d > 0.9) {
                    sum_let += 1.0;
                    int tRight = tIndex;
                    int tLeft = tIndex;
                    int tUp = tIndex;
                    int tDown = tIndex;
                    float pixRes;
                    if(t > 0.9)
                        pixRes = 1.0; 
                    else if (target[++tRight] > 0.9 ||
                             target[--tLeft] > 0.9 ||
                             target[tUp -= tWidth] > 0.9 ||
                             target[tDown += tWidth] > 0.9)
                        pixRes = 0.8;
                    else if (target[++tRight] > 0.9 ||
                             target[--tLeft] > 0.9 ||
                             target[tUp -= tWidth] > 0.9 ||
                             target[tDown += tWidth] > 0.9)
                        pixRes = 0.2;
                    else if (target[++tRight] > 0.9 ||
                             target[--tLeft] > 0.9 ||
                             target[tUp -= tWidth] > 0.9 ||
                             target[tDown += tWidth] > 0.9)
                        pixRes = 0.0;
                    else
                        pixRes = -0.0;
                    sum_conv += pixRes;
                  /*  if(tx_c == 41 && ty_c == 43) {
                        printf("matched pixel (%d,%d) at %f\n", tx, ty, pixRes);
                    } */
                }
#endif
                 
            }
        }
        //float val = (float)sum_conv;
        float val = (float)(sum_conv / sum_let);
        if(val < 0.0)
            val = 0.0;
        if(val > maxVal) {
            maxVal = val;
        }
    }
    results[locId] = maxVal;
}


__global__ void
chartest_kernel(float* distortions, int numDistortions, int maxDistortionSize, float* target, int tWidth, int tHeight, int numLocations, float* map) {
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
#ifdef v1
                sum_let += d;
                sum_conv += (d * t);
#endif
                
                /* Version 2
                 *   rewards matching, punishes noise and missing */
#ifdef v2
                sum_let += d;
                float match = d*t;
                float noise = (1-d)*t;
                float missing = d*(1-t);
                sum_conv += (match - 0.3*noise - 0.7*missing);
#endif

                /* Version 3
                 *   closeness of nearest pixel */
#ifdef v3
                if(d > 0.9) {
                    sum_let += 1.0;
                    int tRight = tIndex;
                    int tLeft = tIndex;
                    int tUp = tIndex;
                    int tDown = tIndex;
                    float pixRes;
                    if(t > 0.9)
                        pixRes = 1.0; 
                    else if (target[++tRight] > 0.9 ||
                             target[--tLeft] > 0.9 ||
                             target[tUp -= tWidth] > 0.9 ||
                             target[tDown += tWidth] > 0.9)
                        pixRes = 0.8;
                    else if (target[++tRight] > 0.9 ||
                             target[--tLeft] > 0.9 ||
                             target[tUp -= tWidth] > 0.9 ||
                             target[tDown += tWidth] > 0.9)
                        pixRes = 0.2;
                    else if (target[++tRight] > 0.9 ||
                             target[--tLeft] > 0.9 ||
                             target[tUp -= tWidth] > 0.9 ||
                             target[tDown += tWidth] > 0.9)
                        pixRes = 0.0;
                    else
                        pixRes = -0.0;
                    sum_conv += pixRes;
                  /*  if(tx_c == 41 && ty_c == 43) {
                        printf("matched pixel (%d,%d) at %f\n", tx, ty, pixRes);
                    } */
                }
#endif
                 
            }
        }
        //float val = (float)sum_conv;
        float val = (float)(sum_conv / sum_let);
        if(val < 0.0)
            val = 0.0;
        if(val > maxVal) {
            maxVal = val;
        }
    }
    map[locId] = maxVal;

//    if(blockId == 0)
//        printf("[T%d found max of %f for (%d,%d) ]\n", threadId, maxVal, locX, locY);
    //__syncthreads();

}

/************************************
 * Reduce Columns Functions
 ***********************************/

void reduce_columns_sequential(int rangeW, int rangeH, int row, float* map, float* results) {
    float max = 0.0;
    for(int c=0; c < rangeH; c++) {
        int mapIndex = (rangeW * c) + row;
        float thisV = map[mapIndex];
        max = (thisV>max)?thisV:max;
    }
    results[row] = max;
}

__global__ void
reduce_columns_kernel(int rangeW, int rangeH, float* map, float* results) {
    int row = threadIdx.x;
    float max = 0.0;
    for(int c=0; c < rangeH; c++) {
        int mapIndex = (rangeW * c) + row;
        float thisV = map[mapIndex];
        max = (thisV>max)?thisV:max;
    }
    results[row] = max;
}

/************************************
 * CharTest Functions
 ***********************************/

double
charTestSequential(float * distortionsBuf, int numDistortions, int maxDistortionSize, float * targetBuf, int targetW, int targetH, int rangeW, int rangeH, float * resultBuf) {

    const int threadsPerBlock = NUM_THREADS_PER_BLOCK;
    int numLocations = rangeW * rangeH;
    const int blocks = ((numLocations + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK);
    
    const int mapBytes = numLocations * sizeof(float);
    float * mapBuf = (float*)malloc(mapBytes);

    double kernelStartTime = CycleTimer::currentSeconds();
    for(int b = 0; b < blocks; b++) {
        printf("Block %d/%d\n", b, blocks);
        for(int t = 0; t < threadsPerBlock; t++) {
            chartest_kernel_sequential(distortionsBuf, numDistortions, maxDistortionSize, targetBuf, targetW, targetH, numLocations, mapBuf, b, t);
        }
    }
    double kernelEndTime = CycleTimer::currentSeconds();

    for(int r = 0; r < rangeW; r++) {
        reduce_columns_sequential(rangeW, rangeH, r, mapBuf, resultBuf);
    }

    free(mapBuf);
    
    return( kernelEndTime - kernelStartTime);

}

double
charTest(float * distortionsBuf, int numDistortions, int maxDistortionSize, float * targetBuf, int targetW, int targetH, int rangeW, int rangeH, float * resultBuf) {
    

    const int targetBytes = targetW * targetH * sizeof(float);
    const int numLocations = rangeW * rangeH;

    // compute number of blocks and threads per block
    const int threadsPerBlock = NUM_THREADS_PER_BLOCK;
    const int blocks = ((numLocations + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK);

    const int mapBytes = numLocations * sizeof(float);
    const int resultBytes = rangeW * sizeof(float);

    // allocate target buffer
    float* device_target;
    cudaMalloc(&device_target, targetBytes);

    // allocate letter buffers
    int distortionBytes = numDistortions * maxDistortionSize * sizeof(float);
    float * device_distortions;
    cudaMalloc( &device_distortions, distortionBytes );

    // allocate results buffers (full map, and final column-reduced)
    float * device_map;
    cudaMalloc( &device_map, mapBytes );
    float * device_result;
    cudaMalloc( &device_result, resultBytes );


    // copy target buffer
    cudaMemcpy(device_target, targetBuf, targetBytes, cudaMemcpyHostToDevice);
    // copy letter buffers
    cudaMemcpy(device_distortions, distortionsBuf, distortionBytes, cudaMemcpyHostToDevice);

    double kernelStartTime = CycleTimer::currentSeconds();
    // run map evaluation kernel
    chartest_kernel<<<blocks, threadsPerBlock>>>(device_distortions, numDistortions, maxDistortionSize, device_target, targetW, targetH, numLocations, device_map);
    cudaThreadSynchronize();
    double kernelEndTime = CycleTimer::currentSeconds();

    // reduce columns
    reduce_columns_kernel<<<1, rangeW>>>(rangeW, rangeH, device_map, device_result);

    // copy result from GPU using cudaMemcpy
    cudaMemcpy( resultBuf, device_result, resultBytes, cudaMemcpyDeviceToHost);
    
    //  free memory buffers on the GPU
    cudaFree(device_target);
    cudaFree(device_distortions);
    cudaFree(device_result);
    
    return( kernelEndTime - kernelStartTime);
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
