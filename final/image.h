#ifndef _IMAGE_H_
#define _IMAGE_H_

#define MAX_PIXELS 50000

typedef struct Image_struct{
  int width;
  int height;
  float data[MAX_PIXELS];
}Image;

int imageRead(Image * buffer, char * path);
int imageWrite(Image * buffer, char * path);

#endif /* _IMAGE_H_ */
