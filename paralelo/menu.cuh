#ifndef MENU_H_INCLUDED
#define MENU_H_INCLUDED

typedef struct {
    int total;
    char *names[20];
} files;

typedef struct {
    char* filePath;
    char typeAlg;
    const char* DIRIMGIN;
    const char* DIRIMGOUT;
    const char* DIRRES;
    int numMaxLinhas;
    int sharedMemory;
    int async;
    int debug;
    int erro;
} initialParams;

#include "../common/funcao.cuh"

void menu(initialParams* ct, int argc, char **argv);

#endif // MENU_H_INCLUDED
