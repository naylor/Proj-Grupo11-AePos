#ifndef KERNEL_H_INCLUDED
#define KERNEL_H_INCLUDED

#include "../common/funcao.h"

__global__ void kernel(unsigned char* kInput, unsigned char* kOutput, int coluna, int linha, int li, int lf);
__global__ void box_filter_kernel_8u_c1(unsigned char* output,const int width, const int height, const size_t pitch, const int lf, const int li);


#endif // KERNEL_H_INCLUDED
