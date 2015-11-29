#ifndef FUNCAO_H_INCLUDED
#define FUNCAO_H_INCLUDED

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

#include "imagem.cuh"
#include "timer.cuh"

unsigned int rand_interval(unsigned int min, unsigned int max);
void getCommandLineOptions(initialParams* ct, files* f, int argc, char *argv[]);
void cleanMemory(PPMImageParams* imageParams, PPMBlock* bloco, timer* t, initialParams* ct);
void writeFile(initialParams* ct, PPMImageParams* imageParams, timer* t);
files* listDir(const char *dir);
int in_array(char *array[], int size, char *lookfor);

#endif // FUNCAO_H_INCLUDED
