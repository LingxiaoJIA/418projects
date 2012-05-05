#ifndef _IMAGE_H_
#define _IMAGE_H_

typedef struct Image_struct{
  int width;
  int height;
}Image;

// Fills in an image header at the start of the buffer (Image) followed by float data
// Returns the number of bytes written to the buffer
// Example of reading image later: Image * ptr = (Image *) buffer; float * data = (float *) (buffer + sizeof(Image))  
int imageRead(char * buffer, char * path);

// Same as imageRead but mallocs and returns a buffer that is the exact size needed for the image header plus data
int imageReadMalloc(char ** buffer, char * path);

// Write out the image to the given file
int imageWrite(char * buffer, char * path);

#endif /* _IMAGE_H_ */
