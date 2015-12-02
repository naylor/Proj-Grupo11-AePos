#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "host.cuh"
#include "kernel.cuh"

#define BLOCK_DIM 32
#define BLOCK_DEFAULT 512


texture<unsigned char, cudaTextureType2D> tex8u;

//Box Filter Kernel For Gray scale image with 8bit depth
__global__ void box_filter_kernel_8u_c1(unsigned char* output,const int width, const int height, const size_t pitch, const int fWidth, const int fHeight)
{
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int filter_offset_x = fWidth/2;
    const int filter_offset_y = fHeight/2;

    float output_value = 0.0f;
    int cont = 0;
    //Make sure the current thread is inside the image bounds
    if(xIndex<width && yIndex<height)
    {
        //Sum the window pixels
        for(int i= -filter_offset_x; i<=filter_offset_x; i++)
        {
            for(int j=-filter_offset_y; j<=filter_offset_y; j++)
            {
                //No need to worry about Out-Of-Range access. tex2D automatically handles it.
                output_value += tex2D(tex8u,xIndex + i,yIndex + j);
                cont++;
            }
        }

        //Average the output value
        output_value = output_value/cont;

        //Write the averaged value to the output.
        //Transform 2D index to 1D index, because image is actually in linear memory
        int index = yIndex * pitch + xIndex;

        output[index] = static_cast<unsigned char>(output_value);
    }
}


void box_filter_8u_c1(initialParams* ct, PPMImageParams* imageParams, PPMBlock* block, int numBlock, cudaStream_t* streamSmooth)
{

    int linhasIn = block[numBlock].linhasIn;
    int linhasOut = block[numBlock].linhasOut;

    const int width = imageParams->coluna;
    const int height = linhasIn;
    const int widthStep = imageParams->coluna;
    const int filterWidth = 5;
    const int filterHeight = 5;

    unsigned char CPUinput[width*height];
    unsigned char CPUoutput[width*linhasOut];

    printf("Apply Smooth[%d][%s] - li:%d, lf:%d\n",
               numBlock, imageParams->tipo, block[numBlock].li, block[numBlock].lf);

    exit(1);

    for(int t=0; t<width*height; t++)
        CPUinput[t] = block[numBlock].pgmIn[t].gray;


    /*
     * 2D memory is allocated as strided linear memory on GPU.
     * The terminologies "Pitch", "WidthStep", and "Stride" are exactly the same thing.
     * It is the size of a row in bytes.
     * It is not necessary that width = widthStep.
     * Total bytes occupied by the image = widthStep x height.
     */

    //Declare GPU pointer
    unsigned char *GPU_input, *GPU_output;

    //Allocate 2D memory on GPU. Also known as Pitch Linear Memory
    size_t gpu_image_pitch = 0;
    cudaMallocPitch<unsigned char>(&GPU_input,&gpu_image_pitch,width,height);
    cudaMallocPitch<unsigned char>(&GPU_output,&gpu_image_pitch,width,height);

    //Copy data from host to device.
    cudaMemcpy2D(GPU_input,gpu_image_pitch,CPUinput,widthStep,width,height,cudaMemcpyHostToDevice);

    //Bind the image to the texture. Now the kernel will read the input image through the texture cache.
    //Use tex2D function to read the image
    cudaBindTexture2D(NULL,tex8u,GPU_input,width,height,gpu_image_pitch);

    /*
     * Set the behavior of tex2D for out-of-range image reads.
     * cudaAddressModeBorder = Read Zero
     * cudaAddressModeClamp  = Read the nearest border pixel
     * We can skip this step. The default mode is Clamp.
     */
    tex8u.addressMode[0] = tex8u.addressMode[1] = cudaAddressModeBorder;

    /*
     * Specify a block size. 256 threads per block are sufficient.
     * It can be increased, but keep in mind the limitations of the GPU.
     * Older GPUs allow maximum 512 threads per block.
     * Current GPUs allow maximum 1024 threads per block
     */

    dim3 block_size(16,16);

    /*
     * Specify the grid size for the GPU.
     * Make it generalized, so that the size of grid changes according to the input image size
     */

    dim3 grid_size;
    grid_size.x = (width + block_size.x - 1)/block_size.x;  /*< Greater than or equal to image width */
    grid_size.y = (height + block_size.y - 1)/block_size.y; /*< Greater than or equal to image height */

    //Launch the kernel
    box_filter_kernel_8u_c1<<<grid_size,block_size>>>(GPU_output,width,height,gpu_image_pitch,filterWidth,filterHeight);

    //Copy the results back to CPU
    cudaMemcpy2D(CPUoutput,widthStep,GPU_output,gpu_image_pitch,width,height,cudaMemcpyDeviceToHost);

    for(int t=0; t<width*linhasOut; t++)
        block[numBlock].pgmOut[t].gray = CPUoutput[t];

    //Release the texture
    cudaUnbindTexture(tex8u);

    //Free GPU memory
    cudaFree(GPU_input);
    cudaFree(GPU_output);
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
        if (ct->sharedMemory == 1)
            blockDims.x = BLOCK_DIM;
        dim3 gridDims((unsigned int) ceil((double)(linhasIn/blockDims.x)), 1, 1 );

        // EXECUTA O CUDAMEMCPY
        // ASSINCRONO OU SINCRONO
        if (ct->async == 1)
            cudaMemcpyAsync( kInput, block[numBlock].ppmIn, linhasIn, cudaMemcpyHostToDevice, streamSmooth[numBlock] );
        else
            cudaMemcpy( kInput, block[numBlock].ppmIn, linhasIn, cudaMemcpyHostToDevice);

        // EXECUTA A FUNCAO SMOOTH NO KERNEL
        // SE A OPCAO DE SHARED MEMORY FOR ATIVADA
        // CHAMA A FUNCAO smoothPPM_SH
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

        // RETORNA A IMAGEM PARA
        // A VARIAVEL DE SAIDA PARA
        // GRAVACAO NO ARQUIVO
        if (ct->async == 1)
            cudaMemcpyAsync(block[numBlock].ppmOut, kOutput, linhasOut, cudaMemcpyDeviceToHost, streamSmooth[numBlock] );
        else
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
