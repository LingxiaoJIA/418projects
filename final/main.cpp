#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <fstream>

#include "image.h"
#include "CycleTimer.h"
#include "defines.h"

// CUDA helper functions
double charTest(char * distortionsBuf, int numDistortions, int maxDistortionBytes, char * device_target, int targetW, int targetH, int rangeW, int rangeH, float* resultBuf);
char * sendTarget(char * targetBuf, int targetBytes);
void freeTarget(char* device_target);

// misc functions
double charTestSequential(float * distortionsBuf, int numDistortions, int maxDistortionSize, float * targetBuf, int targetW, int targetH, int rangeW, int rangeH, float* resultBuf);
void printCudaInfo();
char charLib[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

struct guess {
    char c;
    float val;
} ;

/********************************************
 *  Post processing Helper Functions
 * *****************************************/

void sort(guess * queue) {
    // selection sort
    for(int i=0; i < QUEUE_SIZE-1; i++) {
        // find max above i
        int maxJ = i;
        for(int j = i; j < QUEUE_SIZE; j++)
            if(queue[j].val > queue[maxJ].val)
                maxJ = j;
        // swap max J into spot
        guess temp = queue[i];
        queue[i] = queue[maxJ];
        queue[maxJ] = temp;
    }

}

void processFavor(guess * queue, std::string fav, std::string dis, float amt) {
    for(int i=0; i < 5; i++) {
        char c = queue[i].c;
        size_t first = fav.find(c);
        if(first != std::string::npos) {
            // found a favored
            for(unsigned int j=0; j < i; j++) {
                if(dis.find(queue[j].c) != std::string::npos) {
                    // found a disfavored character above the favored
                    queue[j].val -= amt;
                }
            }
            break;
        }
    }
}

std::string largeLines = "9BDEFHJKLMNPRTUYbdfghjkmnpqrtu";
std::string smallLines = "iIl";
void postProcessL1(guess * queue) {

    /* do v - Y */
    processFavor(queue, std::string("YyMWw"), std::string("vV"), 0.08);
    sort(queue);

    processFavor(queue, std::string("69bdgpqe"), std::string("ou"), 0.04);
    sort(queue);
    
    processFavor(queue, std::string("hmR"), std::string("n"), 0.03);
    sort(queue);
    
    processFavor(queue, std::string("C"), std::string("r"), 0.03);
    sort(queue);
    
    /* do big lines - small lines */
    processFavor(queue, largeLines, smallLines, 0.1);
    sort(queue);
}

void postProcessL2(guess * queue) {
    processFavor(queue, std::string("m"), std::string("en"), 0.07);
    processFavor(queue, std::string("m"), std::string("oa"), 0.03);
    sort(queue);

    processFavor(queue, std::string("nhm"), std::string("r"), 0.1);
    sort(queue);
    
    processFavor(queue, std::string("pdg"), std::string("u"), 0.07);
    
    sort(queue);

}

/********************************************
 *  Misc Helper Functions
 * *****************************************/
        
void printQueue(guess * queue) {
    for(int i=0; i < 6; i++)
        printf("[%c @%.3f]\t", queue[i].c, queue[i].val);
    printf("\n");
}

void printGuess(guess * g, int num) {
    printf("***************************************\n");
    for(int i=0; i < num; i++)
        printf("%c ", g[i].c);
    printf("\n");
}

void usage(const char* progname) {
    printf("Usage: %s [options] captcha_name\n", progname);
    printf("Program Options:\n");
    printf("  -s  Start index\n");
    printf("  -e  End index\n");
    printf("  -l  Distortion Library\n");
    printf("  -p  Post Processing Level (0-2)\n");
    printf("  -?  This message\n");
    exit(-1);
}


int main(int argc, char** argv)
{
    // setup timing stuff
    double startTime = CycleTimer::currentSeconds();
    double kernelDuration = 0.0;

    /************************************
     *  Set Default Options
     ***********************************/
    std::string library = "./lib/mangallowerblurred/";
    //std::string library = "./lib/mangalfullblurred/";
    
    char * targetName;
    int postProcLevel = 0;
    int startIndex = 0;
    int endIndex = 62;
    /************************************
     *  Parse Command Line Options
     ***********************************/
    int c;
    while((c = getopt(argc, argv, "s:e:l:p:")) != EOF) {
        switch (c) {
            case 's':
                startIndex = atoi(optarg);
                break;
            case 'e':
                endIndex = atoi(optarg);
                break;
            case 'l':
                library = std::string(optarg);
                break;
            case 'p':
                postProcLevel = atoi(optarg);
                break;
            default:
                usage(argv[0]);

        }
    }
    if(optind >= argc)
        usage(argv[0]);
    targetName = argv[optind];


    /************************************
     *  Initial Setup
     ***********************************/
    srand((unsigned)time(0));

    int numChars = endIndex - startIndex + 1;
    printf("running from %d to %d @ pp %d\n", startIndex, endIndex, postProcLevel);


    // setup memory stuff
    char * targetBuf;
    int targetBytes = imageReadMalloc(&targetBuf, targetName);
    char * device_target = sendTarget(targetBuf, targetBytes);

    // any sequential processing
    int targetWidth = ((Image *)targetBuf)->width;
    int targetHeight = ((Image *)targetBuf)->height;
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
        std::string charName = std::string(&curChar, 1);
        if((int)curChar >=65 && (int)curChar  <=90)
            charName = charName + "_upper";
        if((int)curChar >=97 && (int)curChar  <=122)
            charName = charName + "_lower";
        //printf("%s\n", charName.c_str());

        std::string libCharPath = library + charName + "/";
        std::string statFile = libCharPath + "stats.txt";
    
        std::ifstream infile(statFile.c_str());
        if(!infile) {
            printf("\tStatefile read %s failed!\n", statFile.c_str());
            return 1;
        }
        int numDistortions, maxDistortionSize;
        infile >> numDistortions;
        infile >> maxDistortionSize;
        infile.close();
        
        printf("Running [%c] @ %d locations x %d distortions \n", curChar, numLocations, numDistortions);

        int maxDistortionBytes = sizeof(Image) + maxDistortionSize * sizeof(float);

        char * distortionsBuf = (char*) malloc(numDistortions * maxDistortionBytes);

        char * thisDistortion = distortionsBuf;
        for(int d = 0; d < numDistortions; d++) {
            char temp[10];
            sprintf(temp, "%d", d);
            std::string distortionPath = library + charName + "/" + temp;
            imageRead(thisDistortion, (char *)distortionPath.c_str());
            thisDistortion += maxDistortionBytes;
        }
        
        /************************************
         *  Setup Results Buffer Output
         ***********************************/
        float* resultBuf = (float*) malloc(rangeWidth * sizeof(float));
        results[charIndex] = resultBuf;

        /************************************
         *  Execute Kernel
         ***********************************/
        printf("Evaluating %c\n", curChar);
    
        kernelDuration += charTest(distortionsBuf, numDistortions, maxDistortionBytes, device_target, targetWidth, targetHeight, rangeWidth, rangeHeight, resultBuf);
        //kernelDuration += charTestSequential(distortionsBuf, numDistortions, maxDistortionSize, device_target, targetWidth, targetHeight, rangeWidth, rangeHeight, resultBuf);
        
        /************************************
         *  Use Results To Guess
         ***********************************/
        
        /************************************
         *  Write Map To Image
         ***********************************/
        //std::string resultPath = "./res/" + charName + "_map";
        //imageWrite( resultBuf, resultPath.c_str(), rangeWidth, rangeHeight);
        
        /************************************
         *  Clean Up For This Letter
         ***********************************/
        free(distortionsBuf);
    }

    freeTarget(device_target);


    /************************************
     *  Post Processing
     ***********************************/
    guess overallGuess[20];

    printf("Beginning post processing\n");
    int gi = 0;
    for(int xmin=0; xmin < targetWidth - WINDOW; xmin += (WINDOW / 2)) {
        int xmax = xmin + WINDOW - 1;
        printf("Window %d - %d\n", xmin, xmax);

        guess queue[QUEUE_SIZE];
        for(int i=0; i<QUEUE_SIZE; i++)
            queue[i].val = 0.0;

        for(int charIndex = startIndex; charIndex < endIndex; charIndex++) {
            char c = charLib[charIndex];
            float maxV = 0.0;
            for(int x = xmin; x < xmax; x++) {
                float v = results[charIndex][x];
                if(v > maxV) {
                    maxV = v;
                }
            }

            // try inserting this value into queue
            int insert = -1;
            for(int i=0; i < QUEUE_SIZE; i++) {
                if(maxV > queue[i].val) {
                    insert = i;
                    break;
                }
            }
            if(insert != -1) {
                for(int i=QUEUE_SIZE-1; i > insert; i--)
                    queue[i] = queue[i-1];
                queue[insert].val = maxV;
                queue[insert].c = c;
            }
        }

        printQueue(queue);

        // clean up queue using post processing (character knowledge)
        if(postProcLevel >= 1)
            postProcessL1(queue);
        if(postProcLevel >= 2)
            postProcessL2(queue);

        printQueue(queue);

        overallGuess[gi++] = queue[0];
    }
    
    /************************************
     *  Final Guess Generation
     ***********************************/
   
    printGuess(overallGuess, gi);
    
    // setup empty final Guess buffer
    guess finalGuess[gi];
    guess empty;
    empty.c = ' ';
    empty.val = 0.0;
    for(int i=0; i <gi; i++) {
        finalGuess[i] = empty;
    }

    // trim edges
    int leftTrim = 0;
    int rightTrim = gi-1;
    while(overallGuess[leftTrim].val < 0.15)
        leftTrim++;
    while(overallGuess[rightTrim].val < 0.15)
        rightTrim--;
    
    // first choose letters who appear twice
    int numChosen = 0;
    for(int i=leftTrim; i < rightTrim; i++) {
        if(overallGuess[i].c == overallGuess[i+1].c) {
            finalGuess[i] = overallGuess[i];
            overallGuess[i] = empty;
            overallGuess[i+1] = empty;
            numChosen++;
        }
    }
    
    printGuess(finalGuess, gi);

    // if got over 6, remove some
    while(numChosen > 6) {
        float minVal = 999999.0;
        int minI = -1;
        for(int i=0; i <gi; i++) {
            float val = finalGuess[i].val;
            if(finalGuess[i].c != ' ' && val < minVal) {
                minVal = val;
                minI = i;
            }
        }        
        printf("removing %d\n", minI);
        overallGuess[minI] = finalGuess[minI];
        finalGuess[minI] = empty;
        numChosen--;
    }
    
    printGuess(finalGuess, gi);

    // if under 6, keep choosing
    int gapStart = leftTrim;
    int gapEnd = gapStart + 1;
    while(numChosen < 6) {
        while(finalGuess[gapEnd].c == ' ' && gapEnd < rightTrim)
            gapEnd++;
        if(gapEnd == rightTrim)
            break;
        gapStart++;
        int gapSize = gapEnd - gapStart;
        if(gapSize >= 3) {
            printf("gap at %d - %d\n", gapStart, gapEnd);
            if(gapSize % 2 == 0) {
                int choice = gapStart + (gapSize / 2) - 1;
                printf("even choose %c or %c\n", overallGuess[choice].c, overallGuess[choice+1].c);
                if(overallGuess[choice].val > overallGuess[choice+1].val) {
                    finalGuess[choice] = overallGuess[choice];
                    overallGuess[choice] = empty;
                    printf("choose %c\n",finalGuess[choice].c);
                } else {
                    finalGuess[choice+1] = overallGuess[choice+1];
                    overallGuess[choice+1] = empty;
                    printf("choose %c\n",finalGuess[choice+1].c);
                }
                    
            } else {
                int choice = gapStart + (gapSize / 2) - 1;
                printf("odd choose %c\n", overallGuess[choice].c);
                finalGuess[choice] = overallGuess[choice];
                overallGuess[choice] = empty;
            }
            
            gapStart++;
            numChosen++;
        } else {
            printf("minigap at %d - %d\n", gapStart, gapEnd);
            gapStart = gapEnd + 1;
            gapEnd = gapStart + 1;
        }

    }



    /************************************
     *  Global Clean Up
     ***********************************/
    free(targetBuf);
    for(int charIndex = startIndex; charIndex < endIndex; charIndex++) {
        free(results[charIndex]);
    }
    
    double endTime = CycleTimer::currentSeconds();
    
    double overallDuration = endTime - startTime;
    printf("************************************\n");
    printf("\tKernel : %.3f ms\n", 1000.f * kernelDuration);
    printf("\tOverall: %.3f ms\n", 1000.f * overallDuration);

    printGuess(finalGuess, gi);
    
    return 0;
}
