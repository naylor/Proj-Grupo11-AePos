#ifndef HOST_H_INCLUDED
#define HOST_H_INCLUDED

#include "../common/imagem.cuh"
#include "kernel.cuh"

void applySmooth(initialParams* ct, PPMImageParams* imageParams,
                 PPMBlock* block, int numBlock, cudaStream_t* streamSmooth);

#endif // HOST_H_INCLUDED
