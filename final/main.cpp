#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <fstream>

#include "defines.h"

void charTest(float * distortionsBuf, int numDistortions, int maxDistortionSize, float * targetBuf, int targetW, int targetH, int numLocations, float* resultBuf);
void printCudaInfo();
float* imageRead(float* buf, int* width, int* height, std::string fileName);
float* imageMallocRead(const char* fileName, int* width, int* height);
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

float* imageRead(float* buf, int * width, int* height, const char* fileName) {
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
    
    readBuf[0] = (float)(w);
    readBuf[1] = (float)(h);
    

    int i = 2;
    while(infile >> hold && i < w * h) {
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

float* imageMallocRead(const char* fileName, int *width, int *height) {
    return imageRead(NULL, width, height, fileName);
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
    printf("running from %d to %d\n", startIndex, endIndex);


    // setup memory stuff
    int targetWidth, targetHeight;
    float * targetBuf = imageMallocRead(targetName, &targetWidth, &targetHeight);

    // any sequential processing
    int rangeWidth = targetWidth - EDGE_DONT_BOTHER;  // dont both with some of the edges
    int rangeHeight = targetHeight - EDGE_DONT_BOTHER;  // dont both with some of the edges
    int numLocations = rangeWidth * rangeHeight;
    
    printCudaInfo();

    char guess[10];
    float guessVal[10];
    for(int g = 0; g < 10; g++) {
        guess[g] = '_';
        guessVal[g] = 0.0;
    }

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
            imageRead(thisDistortion, NULL, NULL, distortionPath.c_str());
            thisDistortion += maxDistortionSize;
        }
        
        /************************************
         *  Setup Results Buffer Output
         ***********************************/
        float* resultBuf = (float*) malloc(numLocations * sizeof(float));

        /************************************
         *  Execute Kernel
         ***********************************/
        printf("Evaluating %c\n", curChar);
    
        charTest(distortionsBuf, numDistortions, maxDistortionSize, targetBuf, targetWidth, targetHeight, numLocations, resultBuf);
        
        /************************************
         *  Use Results To Guess
         ***********************************/
        float maxVal = 0.0;
        int maxLoc = 0;
        for(int r = 0; r < numLocations; r++) {
            if(resultBuf[r] > maxVal) {
                maxVal = resultBuf[r];
                maxLoc = r;
            }
        }
        printf("\tMax Val : %f\n", maxVal);
        if(maxVal > 0.3) {
            int guessX = (maxLoc % rangeWidth) * 10 / rangeWidth;
            if(maxVal > guessVal[guessX]) {
                guess[guessX] = curChar;
                guessVal[guessX] = maxVal;
             }
        }
        
        /************************************
         *  Write Map To Image
         ***********************************/
        std::string resultPath = "./res/" + std::string(&curChar, 1) + "_map";
        imageWrite( resultBuf, resultPath.c_str(), rangeWidth, rangeHeight);
        
        /************************************
         *  Clean Up For This Letter
         ***********************************/
        free(distortionsBuf);
        free(resultBuf);
    }

    printf("Guess : ");
    for(int g=0; g< 10; g++)
        if (guess[g] != '_')
            printf("%c ", guess[g]);
    printf("\n");


    free(targetBuf);
    
    return 0;
}
