#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "kernel.cuh"

#define BLOCK_DIM 32

// FUNCAO PARA APLICAR SMOOTH
// COM SHARED MEMORY EM IMAGENS PGM
__global__ void smoothPGM_SH(PGMPixel* kInput, PGMPixel* kOutput, int coluna, int linha, int li, int lf) {

    // DEFINICAO DO TAMANHO ODO BLOCO PARA
    // MEMORIA COMPARTILHADA
    // RESERVA TECNICA DE 4X4 PARA BORDA
    __shared__ PGMPixel sharedMem[BLOCK_DIM+4][BLOCK_DIM+4];

    // OFFSET DA COLUNA*LINHA
    unsigned int offset = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int c = offset % coluna; // COLUNA
    int l = (offset-c)/coluna; // LINHA

    // TIRANDO A BORDA DO PROCESSAMENTO
    if ( l > lf-li || c < 2 || c > coluna-2 || (li == 0 && l < 2) || (lf==linha-1 && l > (lf-li)-2) )
        return;

    // DEFININDO THREAD+2
    // PARA COMECAR EM -2 (BORDA)
    unsigned int shY = threadIdx.y + 2;
    unsigned int shX = threadIdx.x + 2;

    // POPULANDO O BLOCO 20X20 (4X4 BORDA)
    for(int l = -2; l <= BLOCK_DIM+2; ++l) {
        for(int c = -2; c <= BLOCK_DIM+2; ++c) {
            const int p = (l+offset)+c;
            sharedMem[shY+l][shX+c] = kInput[p];
        }
    }
    // SINCRONIZANDO AS THREADS
    __syncthreads();

    // APLICANDO O SMOOTH NO BLOCO
    float sumg;
    for(int i = -2; i <= 2; ++i)
        for(int j = -2; j <= 2; ++j)
            sumg += sharedMem[shY+i][shX+j].gray;

    // GRAVANDO O RESULTADO
    // NA IMAGEM DE SAIDA
    kOutput[offset].gray = sumg/25;

}

// FUNCAO PARA APLICAR SMOOTH
// COM SHARED MEMORY EM IMAGENS PPM
__global__ void smoothPPM_SH(PPMPixel* kInput, PPMPixel* kOutput, int coluna, int linha, int li, int lf) {
    // DEFINICAO DO TAMANHO ODO BLOCO PARA
    // MEMORIA COMPARTILHADA
    // RESERVA TECNICA DE 4X4 PARA BORDA
    __shared__ PPMPixel sharedMem[BLOCK_DIM+4][BLOCK_DIM+4];

    // OFFSET DA COLUNA*LINHA
    unsigned int offset = blockIdx.x * BLOCK_DIM + threadIdx.x;
    int c = offset % coluna; // COLUNA
    int l = (offset-c)/coluna; // LINHA

    // TIRANDO A BORDA DO PROCESSAMENTO
    if ( l > lf-li || c < 2 || c > coluna-2 || (li == 0 && l < 2) || (lf==linha-1 && l > (lf-li)-2) )
        return;

    // DEFININDO THREAD+2
    // PARA COMECAR EM -2 (BORDA)
    unsigned int shY = threadIdx.y + 2;
    unsigned int shX = threadIdx.x + 2;

    // POPULANDO O BLOCO 20X20 (4X4 BORDA)
    if (threadIdx.x==0 || threadIdx.x==BLOCK_DIM-1 || threadIdx.y==0 || threadIdx.y==BLOCK_DIM-1){
    for(int l = -2; l <= BLOCK_DIM+2; ++l) {
        for(int c = -2; c <= BLOCK_DIM+2; ++c) {
            const int p = (l+offset)+c;
            sharedMem[shY+l][shX+c] = kInput[p];
        }
    }
    }
    // SINCRONIZANDO AS THREADS
    __syncthreads();

    // APLICANDO O SMOOTH NO BLOCO
    float blue;
    float green;
    float red;

    for(int i = -2; i <= 2; ++i) {
        for(int j = -2; j <= 2; ++j) {
            blue += sharedMem[shY+i][shX+j].blue;
            green += sharedMem[shY+i][shX+j].green;
            red += sharedMem[shY+i][shX+j].red;
        }
    }

    // GRAVANDO O RESULTADO
    // NA IMAGEM DE SAIDA
    kOutput[offset].blue = blue/25;
    kOutput[offset].green = green/25;
    kOutput[offset].red = red/25;

}

// FUNCAO PARA APLICAR SMOOTH
// SEM SHARED MEMORY EM IMAGENS PPM
__global__ void smoothPPM_noSH(PPMPixel* kInput, PPMPixel* kOutput, int coluna, int linha, int li, int lf) {

    // OFFSET DA COLUNA*LINHA
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;

    int c = offset % coluna; // COLUNA
    int l = (offset-c)/coluna; // LINHA

    // TIRANDO A BORDA DO PROCESSAMENTO
    if ( l > lf-li || c < 2 || c > coluna-2 || (li == 0 && l < 2) || (lf==linha-1 && l > (lf-li)-2) )
        return;

    // APLICANDO O SMOOTH NO BLOCO
    int sumr = 0;
    int sumb = 0;
    int sumg = 0;

    for(int l2 = -2; l2 <= 2; ++l2) {
        for(int c2 = -2; c2 <= 2; ++c2) {
            if((c+l2) >= 2 && (c+l2) < coluna-2 && (l+c2) >= -2 && (l+c2) <= lf-li+4) {
                int p = (offset + 2*coluna)+(l2*coluna)+c2;
                if (li == 0)
                    p = offset + 2*coluna;
                sumr += kInput[p].red;
                sumg += kInput[p].green;
                sumb += kInput[p].blue;
            }
        }
    }

    // GRAVANDO O RESULTADO
    // NA IMAGEM DE SAIDA
    kOutput[offset].red = sumr/25;
    kOutput[offset].green = sumg/25;
    kOutput[offset].blue = sumb/25;

}

// FUNCAO PARA APLICAR SMOOTH
// SEM SHARED MEMORY EM IMAGENS PGM
__global__ void smoothPGM_noSH(PGMPixel* kInput, PGMPixel* kOutput, int coluna, int linha, int li, int lf) {

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
                sumg += kInput[p].gray;
            }
        }
    }

    // GRAVANDO O RESULTADO
    // NA IMAGEM DE SAIDA
    kOutput[offset].gray = sumg/25;
}
