#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <fstream>

#include "defines.h"

void charTest(float * distortionsBuf, int numDistortions, int maxDistortionSize, float * targetBuf, int targetW, int targetH, int numLocations, float* resultBuf);
void printCudaInfo();
float* imageRead(float* buf, int* width, int* height, std::string fileName, bool);
float* imageMallocRead(const char* fileName, int* width, int* height, bool);
char charLib[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

// return GB/s                                                                                   
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}

void imageWrite(float * buf, const char* fileName, int width, int height) {
    std::ofstream outfile(fileName);
    if(!outfile) {
        printf("\tImage read failed!\n");
        return;
    }
    
    outfile << width << "\n" << height << "\n";

    for(int i=0; i < width * height; i++) {
        int temp = (int)(buf[i] * 255.0);
        outfile << temp << "\n";
    }

    outfile.close();
}

float* imageRead(float* buf, int * width, int* height, const char* fileName, bool storeWH) {
    //printf("Reading image %s\n", fileName);
    int hold;
    std::ifstream infile(fileName);
    if(!infile) {
        printf("\tImage read failed!\n");
        return NULL;
    }

    int w,h;
    infile >> w;
    infile >> h;
    //printf("Dimensions %d x %d\n", w, h);


    float* readBuf;
    if(buf == NULL) {
        //printf("Mallocing buffer\n");
        readBuf = (float*) malloc( w * h * sizeof(float) );
    } else {
        //printf("Using provided buffer\n");
        readBuf = buf;
    }
    
    int i = 0;
    if(storeWH) {
        readBuf[0] = (float)(w);
        readBuf[1] = (float)(h);
        i = 2;
    }

    while(infile >> hold && i < ((w * h)+2)) {
        readBuf[i] = (float)hold / 255.0;
        i++;
    }
    infile.close();
    //printf("done reading, now just finish\n");
    if(width != NULL && height != NULL) {
        *width = w;
        *height = h;
    }
    //printf("Image read success\n");
    return readBuf;
}

float* imageMallocRead(const char* fileName, int *width, int *height, bool storeWH) {
    return imageRead(NULL, width, height, fileName, storeWH);
}


int main(int argc, char** argv)
{
    srand((unsigned)time(0));

    if(argc != 2 && argc != 4) {
        usage(argv[0]);
        return 1;
    }
    char * targetName = argv[1];
    int startIndex = 0;
    int endIndex = 26;
    if(argc == 4) {
        startIndex = atoi(argv[2]);
        endIndex = atoi(argv[3]);
    }
    int numChars = endIndex - startIndex + 1;
    printf("running from %d to %d\n", startIndex, endIndex);


    // setup memory stuff
    int targetWidth, targetHeight;
    float * targetBuf = imageMallocRead(targetName, &targetWidth, &targetHeight, false);

    // any sequential processing
    int rangeWidth = targetWidth - EDGE_DONT_BOTHER;  // dont both with some of the edges
    int rangeHeight = targetHeight - EDGE_DONT_BOTHER;  // dont both with some of the edges
    int numLocations = rangeWidth * rangeHeight;
    
    printCudaInfo();

    float * results[numChars];
    for(int charIndex = startIndex; charIndex < endIndex; charIndex++) {
        char curChar = charLib[charIndex];

        /************************************
         *  Setup Distortions Buffer Input
         ***********************************/
        std::string libCharPath = "./lib/mangallib/" + std::string(&curChar, 1) + "_lower/";
        std::string statFile = libCharPath + "stats.txt";
    
        std::ifstream infile(statFile.c_str());
        if(!infile) {
            printf("\tImage read failed!\n");
            return 1;
        }
        int numDistortions, maxDistortionSize;
        infile >> numDistortions;
        infile >> maxDistortionSize;
        infile.close();
        
        printf("Running [%c] @ %d locations x %d distortions \n", curChar, numLocations, numDistortions);

        maxDistortionSize += 2;
        int maxDistortionBytes = maxDistortionSize * sizeof(float);

        float * distortionsBuf = (float*) malloc(numDistortions * maxDistortionBytes);

        float * thisDistortion = distortionsBuf;
        for(int d = 0; d < numDistortions; d++) {
            char temp[10];
            sprintf(temp, "%d", d);
            std::string distortionPath = "./lib/mangallib/" + std::string(&curChar, 1) + "_lower/" + temp;
            imageRead(thisDistortion, NULL, NULL, distortionPath.c_str(), true);
            thisDistortion += maxDistortionSize;
        }
        
        /************************************
         *  Setup Results Buffer Output
         ***********************************/
        float* resultBuf = (float*) malloc(numLocations * sizeof(float));
        results[charIndex] = resultBuf;

        /************************************
         *  Execute Kernel
         ***********************************/
        printf("Evaluating %c\n", curChar);
    
        charTest(distortionsBuf, numDistortions, maxDistortionSize, targetBuf, targetWidth, targetHeight, numLocations, resultBuf);
        
        /************************************
         *  Use Results To Guess
         ***********************************/
        
        /************************************
         *  Write Map To Image
         ***********************************/
        std::string resultPath = "./res/" + std::string(&curChar, 1) + "_map";
        imageWrite( resultBuf, resultPath.c_str(), rangeWidth, rangeHeight);
        
        /************************************
         *  Clean Up For This Letter
         ***********************************/
        free(distortionsBuf);
    }


    /************************************
     *  Post Processing
     ***********************************/
    printf("Beginning post processing\n");
    for(int xmin=0; xmin < targetWidth - WINDOW; xmin += (WINDOW / 2)) {
        int xmax = x + WINDOW - 1;
        printf("Window %d - %d\n", xmin, xmax);

        char queue[5];
        int size = 0;
        for(int charIndex = startIndex; charIndex < endIndex; charIndex++) {
            int maxV = 0.0;
            for(int y = 0; y < targetHeight; y++) {
                for(int x = xmin; x < xmax; x++) {
                    float v = results[charIndex][y][x];
                    if(v > maxV) {
                        maxV = v;
                    }
                }
            }

            // try inserting this value into queue


            }
        }
    }



    /************************************
     *  Global Clean Up
     ***********************************/
    free(targetBuf);
    for(int charIndex = startIndex; charIndex < endIndex; charIndex++) {
        free(results[charIndex]);
    }

    
    return 0;
}
