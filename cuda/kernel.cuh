#ifndef KERNEL_H_INCLUDED
#define KERNEL_H_INCLUDED

#include "../common/funcao.h"

__global__ void smoothPGM_noSH(PGMPixel* kInput, PGMPixel* kOutput, int coluna, int linha, int li, int lf);
__global__ void smoothPPM_noSH(PPMPixel* kInput, PPMPixel* kOutput, int coluna, int linha, int li, int lf);
__global__ void smoothPPM_SH(PPMPixel* kInput, PPMPixel* kOutput, int coluna, int linha, int li, int lf);
__global__ void smoothPGM_SH(PGMPixel* kInput, PGMPixel* kOutput, int coluna, int linha, int li, int lf);


#endif // KERNEL_H_INCLUDED
