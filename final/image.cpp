#include <stdlib.h>
#include <stdio.h>

#include "image.h"
#include "EasyBMP.h"

int imageRead(char * buffer, char * path){
  int width, height, row, col;
  int red, green, blue;  
  BMP input;
  Image * header = (Image *) buffer;
  float * data = (float *) (buffer + sizeof(Image));

  if (input.ReadFromFile(path) < 0){
    printf("Error reading in file: %s\n", path);
    return -1;
  }
  width = input.TellWidth();
  height = input.TellHeight();

  for(row = 0; row < height; row++){
    for(col = 0; col < width; col++){
      red = input.GetPixel(col, row).Red;
      green = input.GetPixel(col, row).Green;
      blue = input.GetPixel(col, row).Blue;
      data[row*width + col] = 1.0 - (((float) ((red + green + blue)/3)) / 255.0);
    }
  }
  header->width = width;
  header->height = height;
  return (sizeof(Image) + width*height*sizeof(float));
}

int imageReadMalloc(char ** buffer, char * path){
  int width, height, row, col;
  int red, green, blue;
  BMP input;
  int size;
  Image * header;
  float * data;

  if (input.ReadFromFile(path) < 0){
    printf("Error reading in file: %s\n", path);
    return -1;
  }
  width = input.TellWidth();
  height = input.TellHeight();
  size = sizeof(Image) + width*height*sizeof(float);

  *buffer = (char *) malloc(size);
  if(*buffer == NULL) {
    printf("Error mallocing space");
    return -1;
  }
  header = (Image *)(*buffer);
  data = (float *)(*buffer + sizeof(Image));

  for(row = 0; row < height; row++){
    for(col = 0; col < width; col++){
      red = input.GetPixel(col, row).Red;
      green = input.GetPixel(col, row).Green;
      blue = input.GetPixel(col, row).Blue;
      data[row*width + col] = 1.0 - (((float) ((red + green + blue)/3)) / 255.0);
    }
  }
  header->width = width;
  header->height = height;
  return size;
}


int imageWrite(char * buffer, char * path){
  RGBApixel pixel;
  BMP output;
  int row, col;
  Image * header = (Image *) buffer;
  float * data = (float *) (buffer + sizeof(Image));

  output.SetSize(header->width, header->height);

  for(row = 0; row < header->height; row++) {
    for(col = 0; col < header->width; col++){
      pixel.Red = int (data[row*(header->width) + col] * 255);
      pixel.Blue = pixel.Red;
      pixel.Green = pixel.Red;
      pixel.Alpha = 255;
      output.SetPixel(col, row, pixel);
    }
  } 
  
  output.WriteToFile(path);
  return 0;
}
