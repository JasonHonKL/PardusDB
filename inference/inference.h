#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdio.h>
#include <string.h>
// Use const here if the string is not modified
FILE *filereader(const char filename[]);

void closefile(FILE *file);

unsigned short magic_word(FILE *file);

#endif