#include <string>

#include "circleRenderer.h"
#include "cycleTimer.h"
#include "image.h"
#include "ppm.h"

void
startBenchmark(
    CircleRenderer* renderer,
    int startFrame,
    int totalFrames,
    const std::string& frameFilename)
{

    double totalClearTime = 0.f;
    double totalAdvanceTime = 0.f;
    double totalRenderTime = 0.f;
    double totalFileSaveTime = 0.f;
    double totalTime = 0.f;
    double startTime= 0.f;

    bool dumpFrames = frameFilename.length() > 0;

    printf("\nRunning benchmark, %d frames, beginning at frame %d ...\n", totalFrames, startFrame);
    if (dumpFrames)
        printf("Dumping frames to %s_xxx.ppm\n", frameFilename.c_str());

    for (int frame=0; frame<startFrame + totalFrames; frame++) {

        if (frame == startFrame)
            startTime = CycleTimer::currentSeconds();

        double startClearTime = CycleTimer::currentSeconds();

        renderer->clearImage();

        double endClearTime = CycleTimer::currentSeconds();

        renderer->advanceAnimation();

        double endAdvanceTime = CycleTimer::currentSeconds();

        renderer->render();

        double endRenderTime = CycleTimer::currentSeconds();

        if (frame >= startFrame) {
            if (dumpFrames) {
                char filename[1024];
                sprintf(filename, "%s_%04d.ppm", frameFilename.c_str(), frame);
                writePPMImage(renderer->getImage(), filename);
                //renderer->dumpParticles("snow.par");
            }

            double endFileSaveTime = CycleTimer::currentSeconds();

            totalClearTime += endClearTime - startClearTime;
            totalAdvanceTime += endAdvanceTime - endClearTime;
            totalRenderTime += endRenderTime - endAdvanceTime;
            totalFileSaveTime += endFileSaveTime - endRenderTime;
        }
    }

    double endTime = CycleTimer::currentSeconds();
    totalTime = endTime - startTime;

    printf("Clear:    %.4f ms\n", 1000.f * totalClearTime / totalFrames);
    printf("Advance:  %.4f ms\n", 1000.f * totalAdvanceTime / totalFrames);
    printf("Render:   %.4f ms\n", 1000.f * totalRenderTime / totalFrames);
    printf("Total:    %.4f ms\n", 1000.f * (totalClearTime + totalAdvanceTime + totalRenderTime) / totalFrames);
    if (dumpFrames)
        printf("File IO:  %.4f ms\n", 1000.f * totalFileSaveTime / totalFrames);
    printf("\n");
    printf("Overall:  %.4f sec (note units are seconds)\n", totalTime);

}
