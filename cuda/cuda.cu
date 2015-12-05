#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cuda.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

texture<unsigned char, cudaTextureType2D> textureIn;


//Box Filter Kernel For Gray scale image with 8bit depth
__global__ void kernelTexture(unsigned char* output,const int width, const int height, const int lf, const int li)
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
        int index = yIndex + xIndex;
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

void structToArray(PPMImageParams* imageParams, PPMThread* thread,
                   int numThread, unsigned char *cpuIn, int filtro) {

    if (strcmp(imageParams->tipo, "P6")==0) {
        if (filtro == 1)
            for(int t=0; t<thread[numThread].linhasIn * imageParams->coluna; t++)
                cpuIn[t] = thread[numThread].ppmIn[t].red;
        if (filtro == 2)
            for(int t=0; t<thread[numThread].linhasIn * imageParams->coluna; t++)
                cpuIn[t] = thread[numThread].ppmIn[t].green;
        if (filtro == 3)
            for(int t=0; t<thread[numThread].linhasIn * imageParams->coluna; t++)
                cpuIn[t] = thread[numThread].ppmIn[t].blue;
    }

    if (strcmp(imageParams->tipo, "P5")==0) {
        for(int t=0; t<thread[numThread].linhasIn * imageParams->coluna; t++)
            cpuIn[t] = thread[numThread].pgmIn[t].gray;
    }
}

void arrayToStruct(PPMImageParams* imageParams, PPMThread* thread,
                   int numThread, unsigned char* cpuOut, int filtro) {

    if (strcmp(imageParams->tipo, "P6")==0) {
        if (filtro == 1)
            for(int t=0; t<thread[numThread].linhasOut * imageParams->coluna; t++)
                thread[numThread].ppmOut[t].red = cpuOut[t];
        if (filtro == 2)
            for(int t=0; t<thread[numThread].linhasOut * imageParams->coluna; t++)
                thread[numThread].ppmOut[t].green = cpuOut[t];
        if (filtro == 3)
            for(int t=0; t<thread[numThread].linhasOut * imageParams->coluna; t++)
                thread[numThread].ppmOut[t].blue = cpuOut[t];
    }

    if (strcmp(imageParams->tipo, "P5")==0) {
        for(int t=0; t<thread[numThread].linhasOut * imageParams->coluna; t++)
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
    cpuIn = (unsigned char *)malloc(thread[numThread].linhasIn * imageParams->coluna * sizeof(unsigned char) );
    cpuOut = (unsigned char *)malloc(thread[numThread].linhasOut * imageParams->coluna * sizeof(unsigned char) );

    structToArray(imageParams, thread, numThread, cpuIn, filtro);

    //Allocate 2D memory on GPU. Also known as Pitch Linear Memory
    size_t pitch = 0;
    gpuErrchk( cudaMallocPitch<unsigned char>(&gpuIn,&pitch,imageParams->coluna,thread[numThread].linhasIn) );
    gpuErrchk( cudaMallocPitch<unsigned char>(&gpuOut,&pitch,imageParams->coluna,thread[numThread].linhasOut) );


    //Copy data from host to device.
    gpuErrchk( cudaMemcpy2DAsync(gpuIn,pitch,cpuIn,imageParams->coluna,imageParams->coluna,thread[numThread].linhasIn,cudaMemcpyHostToDevice, streamSmooth[numThread]) );

    //Bind the image to the texture. Now the kernel will read the input image through the texture cache.
    //Use tex2D function to read the image
    gpuErrchk( cudaBindTexture2D(NULL,textureIn,gpuIn,imageParams->coluna,thread[numThread].linhasIn,pitch) );

    dim3 blockDims(16,16);
    dim3 gridDims;
    gridDims.x = (imageParams->coluna + blockDims.x - 1)/blockDims.x;
    gridDims.y = (thread[numThread].linhasIn + blockDims.y - 1)/blockDims.y;

    cudaEventRecord(start, 0);
    kernelTexture<<<gridDims,blockDims, 0, streamSmooth[numThread]>>>(gpuOut,imageParams->coluna,imageParams->linha,thread[numThread].lf,thread[numThread].li);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    //Copy the results back to CPU
    cudaMemcpy2DAsync(cpuOut,imageParams->coluna,gpuOut,pitch,imageParams->coluna,thread[numThread].linhasOut,cudaMemcpyDeviceToHost, streamSmooth[numThread]);

    arrayToStruct(imageParams, thread, numThread, cpuOut, filtro);

    //Release the texture
    cudaUnbindTexture(textureIn);

    //Free GPU memory
    cudaFree(gpuIn);
    cudaFree(gpuOut);
    free(cpuIn);
    free(cpuOut);

    cudaEventElapsedTime(&time, start, stop);

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
    cpuIn = (unsigned char *)malloc(thread[numThread].linhasIn * imageParams->coluna * sizeof(unsigned char) );
    cpuOut = (unsigned char *)malloc(thread[numThread].linhasOut * imageParams->coluna * sizeof(unsigned char) );

    structToArray(imageParams, thread, numThread, cpuIn, filtro);

    // ALOCAR MEMORIA
    cudaMalloc( (void**) &gpuIn, thread[numThread].linhasIn * imageParams->coluna);
    cudaMalloc( (void**) &gpuOut, thread[numThread].linhasOut * imageParams->coluna);

    // DEFINICAO DO TAMANHO PADRAO
    // DO BLOCO
    dim3 blockDims(512,1,1);
    dim3 gridDims((unsigned int) ceil((double)(thread[numThread].linhasIn * imageParams->coluna/blockDims.x)), 1, 1 );

    cudaEventRecord(start, 0);
    cudaMemcpyAsync( gpuIn, cpuIn, thread[numThread].linhasIn * imageParams->coluna, cudaMemcpyHostToDevice, streamSmooth[numThread] );
    kernel<<<gridDims, blockDims, 0, streamSmooth[numThread]>>>(gpuIn, gpuOut, imageParams->coluna, imageParams->linha, thread[numThread].li, thread[numThread].lf);
    cudaMemcpyAsync(cpuOut, gpuOut, thread[numThread].linhasOut * imageParams->coluna, cudaMemcpyDeviceToHost, streamSmooth[numThread] );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);


    arrayToStruct(imageParams, thread, numThread, cpuOut, filtro);

    // LIBERA A MEMORIA
    cudaFree(gpuIn);
    cudaFree(gpuOut);
    free(cpuIn);
    free(cpuOut);

    cudaDeviceSynchronize();

    cudaEventElapsedTime(&time, start, stop);

    if (ct->debug >= 1)
        printf("Apply Smooth[%d][%s] - li:%d, lf:%d\n",
               numThread, imageParams->tipo, thread[numThread].li, thread[numThread].lf);

    return time;
}


