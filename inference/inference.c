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

unsigned short magic_word(FILE *file){
    char magic[5];
    fread(magic ,sizeof(char) ,4 , file);
    magic[4] = '\0'; 
    if (strcmp(magic , "GGUF")==0){
        return 1;
    }
    return 0;
}