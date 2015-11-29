#ifndef FUNCAO_H_INCLUDED
#define FUNCAO_H_INCLUDED

#include "../common/imagem.cuh"
#include "../common/timer.cuh"

unsigned int rand_interval(unsigned int min, unsigned int max);
void getCommandLineOptions(initialParams* ct, files* f, int argc, char *argv[]);
files* listDir(const char *dir);
int in_array(char *array[], int size, char *lookfor);
void cleanMemory(initialParams* ct, PPMImageParams* imageParams, PPMBlock* bloco);
void writeFile(initialParams* ct, PPMImageParams* imageParams, timer* tr, timer* tw, timer* ta);

#endif // FUNCAO_H_INCLUDED
