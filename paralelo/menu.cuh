#ifndef MENU_H_INCLUDED
#define MENU_H_INCLUDED

#include "../common/funcao.cuh"

void menu(initialParams* ct, int argc, char **argv);

void box_filter_8u_c1(initialParams* ct, PPMImageParams* imageParams, PPMBlock* block, int numBlock, cudaStream_t* streamSmooth);


#endif // MENU_H_INCLUDED
