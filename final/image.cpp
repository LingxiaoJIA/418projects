#include "image.h"
#include "easybmp/EasyBMP.h"

int imageRead(Image * buffer, char * path){
  int width, height, row, col;
  int red, green, blue;  
  BMP input;

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
      buffer->data[row*width + col] = ((float) ((red + green + blue)/3)) / 255.0;
    }
  }
  buffer->width = width;
  buffer->height = height;
  return 0;
}

int imageWrite(Image * buffer, char * path){
  RGBApixel pixel;
  BMP output;
  int row, col;

  output.SetSize(buffer->width, buffer->height);

  for(row = 0; row < buffer->height; row++) {
    for(col = 0; col < buffer->width; col++){
      pixel.Red = int (buffer->data[row*(buffer->width) + col] * 255);
      pixel.Blue = pixel.Red;
      pixel.Green = pixel.Red;
      pixel.Alpha = 255;
      output.SetPixel(col, row, pixel);
    }
  } 
  
  output.WriteToFile(path);
  return 0;
}
