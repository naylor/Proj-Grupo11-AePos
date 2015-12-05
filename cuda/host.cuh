#ifndef HOST_H_INCLUDED
#define HOST_H_INCLUDED

#include "../common/imagem.h"

float applySmooth(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread,
                 int numThread, cudaStream_t* streamSmooth, int filtro);

float applySmoothTexture(initialParams* ct, PPMImageParams* imageParams,
                       PPMThread* thread, int numThread, cudaStream_t* streamSmooth, int filtro);

void arrayToStruct(PPMImageParams* imageParams, PPMThread* thread,
                   int numThread, unsigned char* cpuOut, int filtro);

void structToArray(PPMImageParams* imageParams, PPMThread* thread,
                   int numThread, unsigned char* cpuIn, int filtro);

#endif // HOST_H_INCLUDED
