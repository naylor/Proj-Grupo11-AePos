#ifndef HOST_H_INCLUDED
#define HOST_H_INCLUDED

#include "../common/imagem.cuh"

void applySmooth(initialParams* ct, PPMImageParams* imageParams,
                 PPMBlock* block, int numBlock, cudaStream_t* streamSmooth,
                 timer* tempoS, timer* tempoM);

#endif // HOST_H_INCLUDED
