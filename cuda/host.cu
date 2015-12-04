#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "host.cuh"
#include "kernel.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


float box_filter_8u_c1(initialParams* ct, PPMImageParams* imageParams,
                       PPMThread* thread, int numThread, cudaStream_t* streamSmooth, int filtro)
{

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int linhasIn = thread[numThread].linhasIn;

    const int width = imageParams->coluna;
    const int height = (thread[numThread].lf-thread[numThread].li)+1;
    const int widthStep = imageParams->coluna;


    unsigned char* CPUinput;
    unsigned char* CPUoutput;
    CPUinput = (unsigned char *)malloc(linhasIn * sizeof(unsigned char));
        exit(1);

    CPUoutput = (unsigned char *)malloc(width*height * sizeof(unsigned char));


    if (strcmp(imageParams->tipo, "P6")==0) {
        if (filtro == 1)
            for(int t=0; t<linhasIn; t++)
                CPUinput[t] = thread[numThread].ppmIn[t].red;
        if (filtro == 2)
            for(int t=0; t<linhasIn; t++)
                CPUinput[t] = thread[numThread].ppmIn[t].green;
        if (filtro == 3)
            for(int t=0; t<linhasIn; t++)
                CPUinput[t] = thread[numThread].ppmIn[t].blue;
    }

    if (strcmp(imageParams->tipo, "P5")==0) {
        for(int t=0; t<linhasIn; t++)
            CPUinput[t] = thread[numThread].pgmIn[t].gray;
    }


    //Declare GPU pointer
    unsigned char *GPU_input, *GPU_output;

    //Allocate 2D memory on GPU. Also known as Pitch Linear Memory
    size_t gpu_image_pitch = 0;
    gpuErrchk( cudaMallocPitch<unsigned char>(&GPU_input,&gpu_image_pitch,width,thread[numThread].linhas) );
    gpuErrchk( cudaMallocPitch<unsigned char>(&GPU_output,&gpu_image_pitch,width,height) );

    //Copy data from host to device.
    gpuErrchk( cudaMemcpy2DAsync(GPU_input,gpu_image_pitch,CPUinput,widthStep,width,thread[numThread].linhas,cudaMemcpyHostToDevice, streamSmooth[numThread]) );

    //Bind the image to the texture. Now the kernel will read the input image through the texture cache.
    //Use tex2D function to read the image
   gpuErrchk( cudaBindTexture2D(NULL,tex8u,GPU_input,width,thread[numThread].linhas,gpu_image_pitch) );

    /*
     * Set the behavior of tex2D for out-of-range image reads.
     * cudaAddressModeBorder = Read Zero
     * cudaAddressModeClamp  = Read the nearest border pixel
     * We can skip this step. The default mode is Clamp.
     */
    //tex8u.addressMode[0] = tex8u.addressMode[1] = cudaAddressModeBorder;

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

    cudaEventRecord(start, 0);
    box_filter_kernel_8u_c1<<<grid_size,block_size, 0, streamSmooth[numThread]>>>(GPU_output,width,imageParams->linha,gpu_image_pitch,thread[numThread].lf,thread[numThread].li);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    //Copy the results back to CPU
    cudaMemcpy2DAsync(CPUoutput,widthStep,GPU_output,gpu_image_pitch,width,height,cudaMemcpyDeviceToHost, streamSmooth[numThread]);

    if (strcmp(imageParams->tipo, "P6")==0) {
        if (filtro == 1)
            for(int t=0; t<width*height; t++)
                thread[numThread].ppmOut[t].red = CPUoutput[t];
        if (filtro == 2)
            for(int t=0; t<width*height; t++)
                thread[numThread].ppmOut[t].green = CPUoutput[t];
        if (filtro == 3)
            for(int t=0; t<width*height; t++)
                thread[numThread].ppmOut[t].blue = CPUoutput[t];
    }

    if (strcmp(imageParams->tipo, "P5")==0) {
        for(int t=0; t<width*height; t++)
            thread[numThread].pgmOut[t].gray = CPUoutput[t];
    }

    //Release the texture
    cudaUnbindTexture(tex8u);

    //Free GPU memory
    cudaFree(GPU_input);
    cudaFree(GPU_output);

    cudaEventElapsedTime(&time, start, stop);

    return time;
}

// FUNCAO __HOST__
// DEFINICAO DOS PARAMETROS DE CHAMADA DO KERNEL
void applySmooth(initialParams* ct, PPMImageParams* imageParams, PPMThread* block, int numBlock, cudaStream_t* streamSmooth) {

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
        dim3 blockDims(512,1,1);
        dim3 gridDims((unsigned int) ceil((double)(linhasIn/blockDims.x)), 1, 1 );

        // EXECUTA O CUDAMEMCPY
        // ASSINCRONO
        cudaMemcpyAsync( kInput, block[numBlock].ppmIn, linhasIn, cudaMemcpyHostToDevice, streamSmooth[numBlock] );

        // EXECUTA A FUNCAO SMOOTH NO KERNEL
        smoothPPM_noSH<<<gridDims, blockDims, 0, streamSmooth[numBlock]>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);

        // RETORNA A IMAGEM PARA
        // A VARIAVEL DE SAIDA PARA
        // GRAVACAO NO ARQUIVO
        cudaMemcpyAsync(block[numBlock].ppmOut, kOutput, linhasOut, cudaMemcpyDeviceToHost, streamSmooth[numBlock] );

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
        dim3 gridDims((unsigned int) ceil((double)(linhasIn/blockDims.x)), 1, 1 );

        // EXECUTA O CUDAMEMCPY
        cudaMemcpyAsync( kInput, block[numBlock].pgmIn, linhasIn, cudaMemcpyHostToDevice, streamSmooth[numBlock] );

        // EXECUTA A FUNCAO SMOOTH NO KERNEL
        smoothPGM_noSH<<<gridDims, blockDims, 0, streamSmooth[numBlock]>>>(kInput, kOutput, imageParams->coluna, imageParams->linha, block[numBlock].li, block[numBlock].lf);

        // RETORNA A IMAGEM PARA
        // A VARIAVEL DE SAIDA PARA
        // GRAVACAO NO ARQUIVO
        cudaMemcpyAsync(block[numBlock].pgmOut, kOutput, linhasOut, cudaMemcpyDeviceToHost, streamSmooth[numBlock] );

        // LIBERA A MEMORIA
        cudaFree(kInput);
        cudaFree(kOutput);
    }

    cudaDeviceSynchronize();

    if (ct->debug >= 1)
        printf("Apply Smooth[%d][%s] - li:%d, lf:%d\n",
               numBlock, imageParams->tipo, block[numBlock].li, block[numBlock].lf);

}
