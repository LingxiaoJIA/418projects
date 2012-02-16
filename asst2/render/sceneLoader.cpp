
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <functional>

#include "sceneLoader.h"
#include "util.h"

// randomFloat --
//
// return a random floating point value between 0 and 1
static float
randomFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
}


void
loadCircleScene(
    SceneName sceneName,
    int& numCircles,
    float*& position,
    float*& velocity,
    float*& color,
    float*& radius)
{

    if (sceneName == SNOWFLAKES) {

        // 100K circles
        //
        // Circles are sorted in reverse depth order (farthest first).
        // This order must be respected by the renderer for correct
        // transparency rendering.

        numCircles = 100 * 1000;

        position = new float[3 * numCircles];
        velocity = new float[3 * numCircles];
        color = new float[3 * numCircles];
        radius = new float[numCircles];

        srand(0);
        std::vector<float> depths(numCircles);

        for (int i=0; i<numCircles; i++) {
            // most of the circles are farther away from the camera
            depths[i] = CLAMP(powf((static_cast<float>(i) / numCircles), .1f) + (-.05f + .1f * randomFloat()), 0.f, 1.f);
        }

        // sort the depths, and then assign depths to particles
        std::sort(depths.begin(), depths.end(), std::greater<float>());

        const static float kMinSnowRadius = .0075f;

        for (int i=0; i<numCircles; i++) {

            float depth = depths[i];

            float closeSize = .08f;
            float actualSize = closeSize - .0075f + (.015f * randomFloat());
            radius[i] = ((1.f - depth) * actualSize) + (depth * actualSize / 15.f);
            if (depth < .02f)
                radius[i] *= 3.f;
            else if (radius[i] < kMinSnowRadius)
                radius[i] = kMinSnowRadius;

            int index3 = 3 * i;
            position[index3] = randomFloat();
            position[index3+1] = 1.f + radius[i] + 2.f * randomFloat();
            position[index3+2] = depth;

            velocity[index3] = 0.f;
            velocity[index3+1] = 0.f;
            velocity[index3+2] = 0.f;
        }

    } else if (sceneName == CIRCLE_RGB) {

        // simple test scene containing 3 circles. All circles have
        // 50% opacity
        //
        // farthest circle is red.  Middle is green.  Closest is blue.

        numCircles = 3;

        position = new float[3 * numCircles];
        velocity = new float[3 * numCircles];
        color = new float[3 * numCircles];
        radius = new float[numCircles];

        for (int i=0; i<numCircles; i++)
            radius[i] = .3f;

        position[0] = .4f;
        position[1] = .5f;
        position[2] = .75f;
        color[0] = 1.f;
        color[1] = 0.f;
        color[2] = 0.f;

        position[3] = .5f;
        position[4] = .5f;
        position[5] = .5f;
        color[3] = 0.f;
        color[4] = 1.f;
        color[5] = 0.f;

        position[6] = .6f;
        position[7] = .5f;
        position[8] = .25f;
        color[6] = 0.f;
        color[7] = 0.f;
        color[8] = 1.f;

    } else if (sceneName == CIRCLE_TEST) {

        // another test scene containing 15K circles

        numCircles = 15 * 1024;

        position = new float[3 * numCircles];
        velocity = new float[3 * numCircles];
        color = new float[3 * numCircles];
        radius = new float[numCircles];

        srand(0);
        std::vector<float> depths(numCircles);
        for (int i=0; i<numCircles; i++) {
            depths[i] = randomFloat();
        }

        std::sort(depths.begin(), depths.end(),  std::greater<float>());

        for (int i=0; i<numCircles; i++) {

            float depth = depths[i];

            radius[i] = .02f + .06f * randomFloat();

            int index3 = 3 * i;

            position[index3] = randomFloat();
            position[index3+1] = randomFloat();
            position[index3+2] = depth;

            color[index3] = .1f + .9f * randomFloat();
            color[index3+1] = .2f + .5f * randomFloat();
            color[index3+2] = .5f + .5f * randomFloat();
        }
    }

}
