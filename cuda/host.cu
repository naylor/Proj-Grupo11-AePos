#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "host.cuh"
#include "kernel.cuh"

#define BLOCK_DIM 32
#define BLOCK_DEFAULT 512

texture<unsigned char, cudaTextureType2D> tex8u;


// FUNCAO PARA APLICAR SMOOTH
// COM SHARED MEMORY EM IMAGENS PPM
__global__ void smoothPPM_SH(PPMPixel* kInput, unsigned char* output, int coluna, int linha, int li, int lf) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  if(xIndex<coluna-100 && yIndex<linha-100)
    {


    output[xIndex] = tex2D(tex8u,xIndex,yIndex);
    }


}

// FUNCAO __HOST__
// DEFINICAO DOS PARAMETROS DE CHAMADA DO KERNEL
void applySmooth(initialParams* ct, PPMImageParams* imageParams, PPMBlock* block, int numBlock, cudaStream_t* streamSmooth) {

    // DEFINE A QUANTIDADE DE LINHAS DO
    // BLOCO LIDO E DO BLOCO QUE SERA
    // GRAVADO EM DISCO
    double linhasIn = block[numBlock].linhasIn;
    double linhasOut = block[numBlock].linhasOut;

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
        dim3 blockDims(BLOCK_DEFAULT,1,1);
        // SE A OPCAO DE SHARED MEMORY
        // FOR ATIVADA, DEFINE O TAMANHO
        // DO BLOCO PARA 32
        //if (ct->sharedMemory == 1)
        //    blockDims.x = BLOCK_DIM;
        dim3 gridDims((unsigned int) ceil((double)(linhasIn/blockDims.x)), 1, 1 );

            //Declare GPU pointer
            unsigned char *GPU_input, *GPU_output;
            //Allocate 2D memory on GPU. Also known as Pitch Linear Memory
            size_t gpu_image_pitch = 0;

        // EXECUTA O CUDAMEMCPY
        // ASSINCRONO OU SINCRONO
        if (ct->async == 1) {

            cudaMallocPitch<unsigned char>(&GPU_input,&gpu_image_pitch,imageParams->coluna,imageParams->linha);
            cudaMallocPitch<unsigned char>(&GPU_output,&gpu_image_pitch,imageParams->coluna,imageParams->linha);

            //Copy data from host to device.
            cudaMemcpy2D(GPU_input,gpu_image_pitch,block[numBlock].ppmIn,imageParams->coluna,imageParams->coluna,imageParams->linha,cudaMemcpyHostToDevice);

            //Bind the image to the texture. Now the kernel will read the input image through the texture cache.
            //Use tex2D function to read the image
            cudaBindTexture2D(NULL,tex8u,GPU_input,imageParams->coluna,imageParams->linha,gpu_image_pitch);

            /*
             * Set the behavior of tex2D for out-of-range image reads.
             * cudaAddressModeBorder = Read Zero
             * cudaAddressModeClamp  = Read the nearest border pixel
             * We can skip this step. The default mode is Clamp.
             */
            tex8u.addressMode[0] = tex8u.addressMode[1] = cudaAddressModeBorder;
        } else
            cudaMemcpy( kInput, block[numBlock].ppmIn, linhasIn, cudaMemcpyHostToDevice);

        // EXECUTA A FUNCAO SMOOTH NO KERNEL
        // SE A OPCAO DE SHARED MEMORY FOR ATIVADA
        // CHAMA A FUNCAO smoothPPM_SH
        if (ct->async == 1) {
            if (ct->sharedMemory == 1)
                smoothPPM_noSH<<<gridDims, blockDims, 0, streamSmooth[numBlock]>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
            else
                smoothPPM_noSH<<<gridDims, blockDims, 0, streamSmooth[numBlock]>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
        } else {
            if (ct->sharedMemory == 1)
                smoothPPM_SH<<<gridDims, blockDims>>>(kInput, GPU_output, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
            else
                smoothPPM_noSH<<<gridDims, blockDims>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);
        }

        // RETORNA A IMAGEM PARA
        // A VARIAVEL DE SAIDA PARA
        // GRAVACAO NO ARQUIVO
        if (ct->async == 1) {
            //Copy the results back to CPU
            cudaMemcpy2D(block[numBlock].ppmOut,imageParams->coluna,GPU_output,gpu_image_pitch,imageParams->coluna,imageParams->linha,cudaMemcpyDeviceToHost);

            //Release the texture
            cudaUnbindTexture(tex8u);
        } else
            cudaMemcpy(block[numBlock].ppmOut, kOutput, linhasOut, cudaMemcpyDeviceToHost );

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
        dim3 blockDims(BLOCK_DEFAULT,1,1);
        // SE A OPCAO DE SHARED MEMORY
        // FOR ATIVADA, DEFINE O TAMANHO
        // DO BLOCO PARA 32
        if (ct->sharedMemory == 1)
            blockDims.x = BLOCK_DIM;
        dim3 gridDims((unsigned int) ceil((double)(linhasIn/blockDims.x)), 1, 1 );

        // EXECUTA O CUDAMEMCPY
        // ASSINCRONO OU SINCRONO
        if (ct->async == 1)
            cudaMemcpyAsync( kInput, block[numBlock].pgmIn, linhasIn, cudaMemcpyHostToDevice, streamSmooth[numBlock] );
        else
            cudaMemcpy( kInput, block[numBlock].pgmIn, linhasIn, cudaMemcpyHostToDevice);

        // EXECUTA A FUNCAO SMOOTH NO KERNEL
        // SE A OPCAO DE SHARED MEMORY FOR ATIVADA
        // CHAMA A FUNCAO smoothPPM_SH
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

        // RETORNA A IMAGEM PARA
        // A VARIAVEL DE SAIDA PARA
        // GRAVACAO NO ARQUIVO
        if (ct->async == 1)
            cudaMemcpyAsync(block[numBlock].pgmOut, kOutput, linhasOut, cudaMemcpyDeviceToHost, streamSmooth[numBlock] );
        else
            cudaMemcpy(block[numBlock].pgmOut, kOutput, linhasOut, cudaMemcpyDeviceToHost );

        // LIBERA A MEMORIA
        cudaFree(kInput);
        cudaFree(kOutput);
    }

    cudaDeviceSynchronize();

    if (ct->debug >= 1)
        printf("Apply Smooth[%d][%s] - li:%d, lf:%d\n",
               numBlock, imageParams->tipo, block[numBlock].li, block[numBlock].lf);

}
