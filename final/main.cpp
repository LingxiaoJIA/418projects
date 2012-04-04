#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <fstream>

#include "defines.h"

void charTest(float * distortionsBuf, float * targetBuf, int targetW, int targetH, int numLocations, float* resultBuf);
void printCudaInfo();
void imageRead(float* buf, std::string fileName, int width, int height);
float* imageMallocRead(const char* fileName, int width, int height);
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

void imageRead(float* buf, const char* fileName, int width, int height) {
   // printf("Reading in image \"%s\" (%dx%d)\n", fileName, width, height);

    /* fake it */
/*    for(int i=0; i < (width * height); i++) {
        buf[i] = (float) rand() / (float) RAND_MAX;
    } */

    int i = 0;
    int hold;
    std::ifstream infile(fileName);
    if(!infile) {
        printf("\tImage read failed!\n");
        return;
    }

    while(infile >> hold && i < width * height) {
        buf[i] = (float)hold / 255.0;
        i++;
    }
    infile.close();
}

float* imageMallocRead(const char* fileName, int width, int height) {
    int bytes = width * height * sizeof(float);
    float* temp = (float*)malloc(bytes);
    imageRead(temp, fileName, width, height);
    return temp;
}


int main(int argc, char** argv)
{
    srand((unsigned)time(0));

    int startIndex = 0;
    int endIndex = 62;
    if(argc == 3) {
        startIndex = atoi(argv[1]);
        endIndex = atoi(argv[2]);
    } else if (argc != 1) {
        usage(argv[0]);
    }
    printf("running from %d to %d\n", startIndex, endIndex);

    // setup memory stuff
    int targetWidth = 400;
    int targetHeight = 180;
    int rangeWidth = targetWidth - LETTER_WIDTH;
    int rangeHeight = targetHeight - LETTER_HEIGHT;
    int numLocations = rangeWidth * rangeHeight;

    float * targetBuf = imageMallocRead("./img/sampleimg.txt", targetWidth, targetHeight);
    // any sequential processing
    
    printCudaInfo();

    char guess[10];
    for(int g = 0; g < 10; g++)
        guess[g] = '_';

    for(int charIndex = startIndex; charIndex < endIndex; charIndex++) {
        char curChar = charLib[charIndex];
        printf("Running [%c] @ %d locations x %d distortions \n", curChar, numLocations, NUM_DISTORTIONS);

        /************************************
         *  Setup Distortions Buffer Input
         ***********************************/
        float * distortionsBuf = (float*) malloc(DISTORTIONS_BUFFER_BYTES);

        float * thisDistortion = distortionsBuf;
        for(int d = 0; d < NUM_DISTORTIONS; d++) {
            char temp[10];
            sprintf(temp, "%d", d);
//            std::string letterPath = "./letters/" + std::string(&curChar, 1) + "_" + std::string(temp);
            std::string distortionPath = "./img/" + std::string(&curChar, 1) + "/" + temp;
            imageRead(thisDistortion, distortionPath.c_str(), LETTER_WIDTH, LETTER_HEIGHT);
            thisDistortion += LETTER_BYTES;
        }
        
        /************************************
         *  Setup Results Buffer Output
         ***********************************/
        float* resultBuf = (float*) malloc(numLocations * sizeof(float));

        /************************************
         *  Execute Kernel
         ***********************************/
        //printf("Evaluating %c\n", curChar);
    
        charTest(distortionsBuf, targetBuf, targetWidth, targetHeight, numLocations, resultBuf);
        
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
            guess[guessX] = curChar;
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
