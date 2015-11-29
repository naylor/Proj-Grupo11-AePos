#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "host.cuh"
#include "kernel.cuh"

#define BLOCK_DIM 32

// FUNCAO __HOST__
// DEFINICAO DOS PARAMETROS DE CHAMADA DO KERNEL
void applySmooth(initialParams* ct, PPMImageParams* imageParams, PPMBlock* block, int numBlock,
                  cudaStream_t* streamSmooth, timer* tempoS, timer* tempoM) {

    // DEFINE A QUANTIDADE DE LINHAS DO
    // BLOCO LIDO E DO BLOCO QUE SERA
    // GRAVADO EM DISCO
    int linhasIn = block[numBlock].linhasIn;
    int linhasOut = block[numBlock].linhasOut;

    // SE A IMAGEM FOR PPM
    if (strcmp(imageParams->tipo, "P6")==0) {
        // VARIAVEL PARA COPIA DA IMAGEM
        // PARA O KERNEL
        PPMPixel* kInput;
        PPMPixel* kOutput;

        // ALOCAR MEMORIA
        cudaMalloc( (void**) &kInput, linhasIn);
        cudaMalloc( (void**) &kOutput, linhasOut);

        // DEFINICAO DO TAMANHO PADRAO
        // DO BLOCO
        dim3 blockDims(512,1,1);
        // SE A OPCAO DE SHARED MEMORY
        // FOR ATIVADA, DEFINE O TAMANHO
        // DO BLOCO PARA 32
        if (ct->sharedMemory == 1)
            blockDims.x = BLOCK_DIM;
        dim3 gridDims((unsigned int) ceil((double)(linhasIn/blockDims.x)), 1, 1 );

        // EXECUTA O CUDAMEMCPY
        // ASSINCRONO OU SINCRONO
        start_timer(tempoM); //INICIA O RELOGIO
        if (ct->async == 1)
            cudaMemcpyAsync( kInput, block[numBlock].ppmIn, linhasIn, cudaMemcpyHostToDevice, streamSmooth[numBlock] );
        else
            cudaMemcpy( kInput, block[numBlock].ppmIn, linhasIn, cudaMemcpyHostToDevice);
        stop_timer(tempoM); //PARA O RELOGIO

        // EXECUTA A FUNCAO SMOOTH NO KERNEL
        // SE A OPCAO DE SHARED MEMORY FOR ATIVADA
        // CHAMA A FUNCAO smoothPPM_SH
        start_timer(tempoS); //INICIA O RELOGIO
        if (ct->async == 1) {
            if (ct->sharedMemory == 1)
                smoothPPM_SH<<<gridDims, blockDims, 0, streamSmooth[numBlock]>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
            else
                smoothPPM_noSH<<<gridDims, blockDims, 0, streamSmooth[numBlock]>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
        } else {
            if (ct->sharedMemory == 1)
                smoothPPM_SH<<<gridDims, blockDims>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
            else
                smoothPPM_noSH<<<gridDims, blockDims>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
        }
        stop_timer(tempoS); //PARA O RELOGIO

        // RETORNA A IMAGEM PARA
        // A VARIAVEL DE SAIDA PARA
        // GRAVACAO NO ARQUIVO
        start_timer(tempoM); //INICIA O RELOGIO
        if (ct->async == 1)
            cudaMemcpyAsync(block[numBlock].ppmOut, kOutput, linhasOut, cudaMemcpyDeviceToHost, streamSmooth[numBlock] );
        else
            cudaMemcpy(block[numBlock].ppmOut, kOutput, linhasOut, cudaMemcpyDeviceToHost );
        stop_timer(tempoM); //PARA O RELOGIO

        // LIBERA A MEMORIA
        cudaFree(kInput);
        cudaFree(kOutput);
    }

    // SE A IMAGEM FOR PPM
    if (strcmp(imageParams->tipo, "P5")==0) {
        // VARIAVEL PARA COPIA DA IMAGEM
        // PARA O KERNEL
        PGMPixel* kInput;
        PGMPixel* kOutput;

        // ALOCAR MEMORIA
        cudaMalloc( (void**) &kInput, linhasIn);
        cudaMalloc( (void**) &kOutput, linhasOut);

        // DEFINICAO DO TAMANHO PADRAO
        // DO BLOCO
        dim3 blockDims(512,1,1);
        // SE A OPCAO DE SHARED MEMORY
        // FOR ATIVADA, DEFINE O TAMANHO
        // DO BLOCO PARA 32
        if (ct->sharedMemory == 1)
            blockDims.x = BLOCK_DIM;
        dim3 gridDims((unsigned int) ceil((double)(linhasIn/blockDims.x)), 1, 1 );

        // EXECUTA O CUDAMEMCPY
        // ASSINCRONO OU SINCRONO
        start_timer(tempoM); //INICIA O RELOGIO
        if (ct->async == 1)
            cudaMemcpyAsync( kInput, block[numBlock].pgmIn, linhasIn, cudaMemcpyHostToDevice, streamSmooth[numBlock] );
        else
            cudaMemcpy( kInput, block[numBlock].pgmIn, linhasIn, cudaMemcpyHostToDevice);
        stop_timer(tempoM); //PARA O RELOGIO

        // EXECUTA A FUNCAO SMOOTH NO KERNEL
        // SE A OPCAO DE SHARED MEMORY FOR ATIVADA
        // CHAMA A FUNCAO smoothPPM_SH
        start_timer(tempoS); //INICIA O RELOGIO
        if (ct->async == 1) {
            if (ct->sharedMemory == 1)
                smoothPGM_SH<<<gridDims, blockDims, 0, streamSmooth[numBlock]>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
            else
                smoothPGM_noSH<<<gridDims, blockDims, 0, streamSmooth[numBlock]>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
        } else {
            if (ct->sharedMemory == 1)
                smoothPGM_SH<<<gridDims, blockDims>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
            else
                smoothPGM_noSH<<<gridDims, blockDims>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
        }
        stop_timer(tempoS); //PARA O RELOGIO

        // RETORNA A IMAGEM PARA
        // A VARIAVEL DE SAIDA PARA
        // GRAVACAO NO ARQUIVO
        start_timer(tempoM); //INICIA O RELOGIO
        if (ct->async == 1)
            cudaMemcpyAsync(block[numBlock].pgmOut, kOutput, linhasOut, cudaMemcpyDeviceToHost, streamSmooth[numBlock] );
        else
            cudaMemcpy(block[numBlock].pgmOut, kOutput, linhasOut, cudaMemcpyDeviceToHost );
        stop_timer(tempoM); //PARA O RELOGIO

        // LIBERA A MEMORIA
        cudaFree(kInput);
        cudaFree(kOutput);
    }

    cudaDeviceSynchronize();

    if (ct->debug >= 1)
        printf("Apply Smooth[%d][%s] - L[%d] li:%d, lf:%d\n",
               numBlock, imageParams->tipo, linhasIn, block[numBlock].li, block[numBlock].lf);

}
