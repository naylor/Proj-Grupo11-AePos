#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "kernel.cuh"

//Box Filter Kernel For Gray scale image with 8bit depth
__global__ void box_filter_kernel_8u_c1(unsigned char* output,const int width, const int height, const size_t pitch, const int lf, const int li)
{

    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;


    float output_value = 0.0f;
    int cont = 0;

    // TIRANDO A BORDA DO PROCESSAMENTO
    if ( yIndex > lf-li || xIndex < 2 || xIndex > width-2 || (li == 0 && yIndex < 2) || (lf==height-1 && yIndex > (lf-li)-2) )
        return;

    int inicio = 0;
    if (li != 0)
        inicio = 2;

        for(int l2= -2; l2<=2; l2++)
        {
            for(int c2=-2; c2<=2; c2++)
            {
            if(l2 >= 0 && c2 >= 0) {
                output_value += tex2D(tex8u,inicio+ xIndex+l2,yIndex + c2);
                cont++;
            }
            }
        }

        //Average the output value
        output_value = output_value/cont;

        //Write the averaged value to the output.
        //Transform 2D index to 1D index, because image is actually in linear memory
        int index = yIndex * pitch + xIndex;
        //printf("Smooth index:%d, xIndex:%d yIndex %d lf-li %d\n",index, xIndex, yIndex, lf-li);

        output[index] = static_cast<unsigned char>(output_value);

}
