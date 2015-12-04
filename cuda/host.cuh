#ifndef HOST_H_INCLUDED
#define HOST_H_INCLUDED

#include "../common/imagem.h"

void applySmooth(initialParams* ct, PPMImageParams* imageParams,
                 PPMThread* thread, int numThread, cudaStream_t* streamSmooth);

#endif // HOST_H_INCLUDED
