#ifndef MAIN_H_INCLUDED
#define MAIN_H_INCLUDED

float box_filter_8u_c1(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread,
                        int numThread, cudaStream_t* streamSmooth, int filtro);


#endif // MAIN_H_INCLUDED
