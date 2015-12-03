#ifndef MAIN_H_INCLUDED
#define MAIN_H_INCLUDED

double box_filter_8u_c1(initialParams* ct, PPMImageParams* imageParams, PPMBlock* block, int numBlock, cudaStream_t* streamSmooth);


#endif // MAIN_H_INCLUDED
