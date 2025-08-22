#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdio.h>

// Use const here if the string is not modified
FILE *filereader(const char filename[]);

void closefile(FILE *file);

#endif