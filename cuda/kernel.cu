#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "kernel.cuh"

//Box Filter Kernel For Gray scale image with 8bit depth
__global__ void kernelTexture(unsigned char* output,const int width, const int height, const size_t pitch, const int lf, const int li)
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
                output_value += tex2D(textureIn,inicio+ xIndex+l2,yIndex + c2);
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

// FUNCAO PARA APLICAR SMOOTH
// SEM SHARED MEMORY EM IMAGENS PGM
__global__ void kernel(unsigned char* kInput, unsigned char* kOutput, int coluna, int linha, int li, int lf) {

    // OFFSET DA COLUNA*LINHA
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;

    int c = offset % coluna; // COLUNA
    int l = (offset-c)/coluna; // LINHA

    // TIRANDO A BORDA DO PROCESSAMENTO
    if ( l > lf-li || c < 2 || c > coluna-2 || (li == 0 && l < 2) || (lf==linha-1 && l > (lf-li)-2) )
        return;

    // APLICANDO O SMOOTH NO BLOCO
    int sumg = 0;

    for(int l2 = -2; l2 <= 2; ++l2) {
        for(int c2 = -2; c2 <= 2; ++c2) {
            if((c+l2) >= 2 && (c+l2) < coluna-2 && (l+c2) >= -2 && (l+c2) <= lf-li+4) {
                int p = (offset + 2*coluna)+(l2*coluna)+c2;
                if (li == 0)
                    p = offset + 2*coluna;
                sumg += kInput[p];
            }
        }
    }

    // GRAVANDO O RESULTADO
    // NA IMAGEM DE SAIDA
    kOutput[offset] = sumg/25;
}
