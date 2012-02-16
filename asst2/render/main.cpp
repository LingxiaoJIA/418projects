#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "refRenderer.h"
#include "cudaRenderer.h"
#include "platformgl.h"


void startRendererWithDisplay(CircleRenderer* renderer);
void startBenchmark(CircleRenderer* renderer, int startFrame, int totalFrames, const std::string& frameFilename);


void usage(const char* progname) {
    printf("Usage: %s [options] scenename\n", progname);
    printf("Program Options:\n");
    printf("  -b  --bench <START:END>    Benchmark mode, do not create display. Time frames [START,END)\n");
    printf("  -f  --file  <FILENAME>     Dump frames in benchmark mode (FILENAME_xxxx.ppm)\n");
    printf("  -r  --renderer <ref/fast>  Select renderer: ref or cuda\n");
    printf("  -s  --size  <INT>          Make rendered image <INT>x<INT> pixels\n");
    printf("  -?  --help                 This message\n");
}


int main(int argc, char** argv)
{

    int benchmarkFrameStart = -1;
    int benchmarkFrameEnd = -1;
    int imageSize = 768;

    std::string sceneNameStr;
    std::string frameFilename;
    SceneName sceneName;
    bool useRefRenderer = true;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"help",     0, 0,  '?'},
        {"bench",    1, 0,  'b'},
        {"file",     1, 0,  'f'},
        {"renderer", 1, 0,  'r'},
        {"size",     1, 0,  's'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "b:f:r:s:?", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'b':
            if (sscanf(optarg, "%d:%d", &benchmarkFrameStart, &benchmarkFrameEnd) != 2) {
                fprintf(stderr, "Invalid argument to -b option\n");
                usage(argv[0]);
                exit(1);
            }
            break;
        case 'f':
            frameFilename = optarg;
            break;
        case 'r':
            if (std::string(optarg).compare("cuda") == 0) {
                useRefRenderer = false;
            }
            break;
        case 's':
            imageSize = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////


    if (optind + 1 > argc) {
        fprintf(stderr, "Error: missing scene name\n");
        usage(argv[0]);
        return 1;
    }

    sceneNameStr = argv[optind];

    if (sceneNameStr.compare("snow") == 0)
        sceneName = SNOWFLAKES;
    else if (sceneNameStr.compare("rgb") == 0)
        sceneName = CIRCLE_RGB;
    else if (sceneNameStr.compare("circles") == 0)
        sceneName = CIRCLE_TEST;
    else {
        fprintf(stderr, "Unknown scene name (%s). valid names are: rgb circles snow\n", sceneNameStr.c_str());
        return 1;
    }

    printf("Rendering to %dx%d image\n", imageSize, imageSize);

    CircleRenderer* renderer;

    if (useRefRenderer)
        renderer = new RefRenderer();
    else
        renderer = new CudaRenderer();

    renderer->allocOutputImage(imageSize, imageSize);
    renderer->loadScene(sceneName);
    renderer->setup();

    if (benchmarkFrameStart >= 0)
        startBenchmark(renderer, benchmarkFrameStart, benchmarkFrameEnd - benchmarkFrameStart, frameFilename);
    else {
        glutInit(&argc, argv);
        startRendererWithDisplay(renderer);
    }

    return 0;
}
