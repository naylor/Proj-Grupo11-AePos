#ifndef KERNEL_H_INCLUDED
#define KERNEL_H_INCLUDED

#include "../common/imagem.cuh"

__global__ void smoothPGM_noSH(PGMPixel* input_image, PGMPixel* output_image, int coluna, int linha, int li, int lf);
__global__ void smoothPPM_noSH(PPMPixel* input_image, PPMPixel* output_image, int coluna, int linha, int li, int lf);
__global__ void smoothPPM_SH(PPMPixel* input_image, PPMPixel* output_image, int coluna, int linha, int li, int lf);
__global__ void smoothPGM_SH(PGMPixel* input_image, PGMPixel* output_image, int coluna, int linha, int li, int lf);


#endif // KERNEL_H_INCLUDED
