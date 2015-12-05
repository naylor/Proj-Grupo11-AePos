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

void structToArray(PPMImageParams* imageParams, PPMThread* thread,
                   int numThread, unsigned char *cpuIn, int filtro) {

    if (strcmp(imageParams->tipo, "P6")==0) {
        if (filtro == 1)
            for(int t=0; t<thread[numThread].linhas*imageParams->coluna; t++)
                cpuIn[t] = thread[numThread].ppmIn[t].red;
        if (filtro == 2)
            for(int t=0; t<thread[numThread].linhas*imageParams->coluna; t++)
                cpuIn[t] = thread[numThread].ppmIn[t].green;
        if (filtro == 3)
            for(int t=0; t<thread[numThread].linhas*imageParams->coluna; t++)
                cpuIn[t] = thread[numThread].ppmIn[t].blue;
    }

    if (strcmp(imageParams->tipo, "P5")==0) {
        for(int t=0; t<thread[numThread].linhas*imageParams->coluna; t++)
            cpuIn[t] = thread[numThread].pgmIn[t].gray;
    }
}

void arrayToStruct(PPMImageParams* imageParams, PPMThread* thread,
                   int numThread, unsigned char* cpuOut, int filtro) {

    const int linhas = ((thread[numThread].lf-thread[numThread].li)+1)*imageParams->coluna;

    if (strcmp(imageParams->tipo, "P6")==0) {
        if (filtro == 1)
            for(int t=0; t<linhas; t++)
                thread[numThread].ppmOut[t].red = cpuOut[t];
        if (filtro == 2)
            for(int t=0; t<linhas; t++)
                thread[numThread].ppmOut[t].green = cpuOut[t];
        if (filtro == 3)
            for(int t=0; t<linhas; t++)
                thread[numThread].ppmOut[t].blue = cpuOut[t];
    }

    if (strcmp(imageParams->tipo, "P5")==0) {
        for(int t=0; t<linhas; t++)
            thread[numThread].pgmOut[t].gray = cpuOut[t];
    }

}

float applySmoothTexture(initialParams* ct, PPMImageParams* imageParams,
                       PPMThread* thread, int numThread, cudaStream_t* streamSmooth, int filtro) {

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned char *cpuIn, *cpuOut, *gpuIn, *gpuOut;
    cpuIn = (unsigned char *)malloc(thread[numThread].linhasIn);
    cpuOut = (unsigned char *)malloc(thread[numThread].linhasOut);

    int linhas = (thread[numThread].lf-thread[numThread].li)+1;
    const int widthStep = imageParams->coluna;

    structToArray(imageParams, thread, numThread, cpuIn, filtro);

    //Allocate 2D memory on GPU. Also known as Pitch Linear Memory
    size_t gpu_image_pitch = 0;
    gpuErrchk( cudaMallocPitch<unsigned char>(&gpuIn,&gpu_image_pitch,imageParams->coluna,linhas) );
    gpuErrchk( cudaMallocPitch<unsigned char>(&gpuOut,&gpu_image_pitch,imageParams->coluna,linhas) );


    //Copy data from host to device.
    gpuErrchk( cudaMemcpy2DAsync(gpuIn,gpu_image_pitch,cpuIn,widthStep,imageParams->coluna,thread[numThread].linhas,cudaMemcpyHostToDevice, streamSmooth[numThread]) );

    //Bind the image to the texture. Now the kernel will read the input image through the texture cache.
    //Use tex2D function to read the image
    gpuErrchk( cudaBindTexture2D(NULL,textureIn,gpuIn,imageParams->coluna,thread[numThread].linhas,gpu_image_pitch) );

    dim3 blockDims(16,16);
    dim3 gridDims;
    gridDims.x = (imageParams->coluna + blockDims.x - 1)/blockDims.x;
    gridDims.y = (thread[numThread].linhas + blockDims.y - 1)/blockDims.y;

    cudaEventRecord(start, 0);
    kernelTexture<<<gridDims,blockDims, 0, streamSmooth[numThread]>>>(gpuOut,imageParams->coluna,imageParams->linha,gpu_image_pitch,thread[numThread].lf,thread[numThread].li);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    //Copy the results back to CPU
    gpuErrchk( cudaMemcpy2DAsync(cpuOut,widthStep,gpuOut,gpu_image_pitch,imageParams->coluna,linhas,cudaMemcpyDeviceToHost, streamSmooth[numThread]) );

    arrayToStruct(imageParams, thread, numThread, cpuOut, filtro);

    //Release the texture
    cudaUnbindTexture(textureIn);

    //Free GPU memory
    cudaFree(gpuIn);
    cudaFree(gpuOut);
    free(cpuIn);
    free(cpuOut);

    cudaEventElapsedTime(&time, start, stop);

    cudaDeviceSynchronize();

    if (ct->debug >= 1)
        printf("Apply Smooth[%d][%s] - linhas: %d, li:%d, lf:%d\n",
               numThread, imageParams->tipo, thread[numThread].linhasIn, thread[numThread].li, thread[numThread].lf);

    return time;
}

// FUNCAO __HOST__
// DEFINICAO DOS PARAMETROS DE CHAMADA DO KERNEL
float applySmooth(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread,
                 int numThread, cudaStream_t* streamSmooth, int filtro) {

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // DEFINE A QUANTIDADE DE LINHAS DO
    // BLOCO LIDO E DO BLOCO QUE SERA
    // GRAVADO EM DISCO
    unsigned char *cpuIn, *cpuOut, *gpuIn, *gpuOut;
    cpuIn = (unsigned char *)malloc(thread[numThread].linhasIn);
    cpuOut = (unsigned char *)malloc(thread[numThread].linhasOut);

    structToArray(imageParams, thread, numThread, cpuIn, filtro);
    // ALOCAR MEMORIA
    cudaMalloc( (void**) &gpuIn, thread[numThread].linhasIn);
    cudaMalloc( (void**) &gpuOut, thread[numThread].linhasOut);

    // DEFINICAO DO TAMANHO PADRAO
    // DO BLOCO
    dim3 blockDims(512,1,1);
    dim3 gridDims((unsigned int) ceil((double)(thread[numThread].linhasIn/blockDims.x)), 1, 1 );

    // EXECUTA O CUDAMEMCPY
    // ASSINCRONO
    gpuErrchk( cudaMemcpyAsync( gpuIn, cpuIn, thread[numThread].linhasIn, cudaMemcpyHostToDevice, streamSmooth[numThread] ) );

    cudaEventRecord(start, 0);
    kernel<<<gridDims, blockDims, 0, streamSmooth[numThread]>>>(gpuIn, gpuOut, imageParams->coluna, imageParams->linha, thread[numThread].li, thread[numThread].lf);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    gpuErrchk( cudaMemcpyAsync(cpuOut, gpuOut, thread[numThread].linhasOut, cudaMemcpyDeviceToHost, streamSmooth[numThread] ) );

    arrayToStruct(imageParams, thread, numThread, cpuOut, filtro);

    // LIBERA A MEMORIA
    cudaFree(gpuIn);
    cudaFree(gpuOut);
    free(cpuIn);
    free(cpuOut);

    cudaEventElapsedTime(&time, start, stop);

    cudaDeviceSynchronize();

    if (ct->debug >= 1)
        printf("Apply Smooth[%d][%s] - li:%d, lf:%d\n",
               numThread, imageParams->tipo, thread[numThread].li, thread[numThread].lf);

    return time;
}
