#include "image.h"

int main(int argc, char** argv){
  char * buffer;
  imageReadMalloc(&buffer, "test.bmp");
  imageWrite(buffer, "out.bmp");
  return 0;
}
