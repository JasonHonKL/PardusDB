#include "inference.h"

FILE *filereader(const char filename[]) {
    FILE *file = fopen(filename, "r");
    return file;
}

void closefile(FILE *file) {
    if (file != NULL) {
        fclose(file);
    }
}