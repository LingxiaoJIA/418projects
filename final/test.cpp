#include "image.h"

int main(int argc, char** argv){
  Image test;
  imageRead(&test, "test.bmp");
  imageWrite(&test, "out.bmp");
  return 0;
}
