#ifndef HOST_H_INCLUDED
#define HOST_H_INCLUDED

#include "../common/imagem.h"

void applySmooth(initialParams* ct, PPMImageParams* imageParams,
                 PPMThread* thread, int numThread, cudaStream_t* streamSmooth);


float box_filter_8u_c1(initialParams* ct, PPMImageParams* imageParams,
                       PPMThread* thread, int numThread, cudaStream_t* streamSmooth, int filtro);

#endif // HOST_H_INCLUDED
