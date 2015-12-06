#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "cuda.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// FUNCAO PARA VISUALIZAR AS
// MENSAGENS DE ERRO DO SISTEMA
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// MEMORIA TEXTURE
texture<unsigned char, cudaTextureType2D> textureIn;


// FUNCAO KERNEL
// APLICA O SMOOTH UTILIZANDO
// MEMORIA TEXTURE
__global__ void kernelTexture(unsigned char* kOutput,const int coluna, const int linha,
                              const size_t pitch, const int lf, const int li) {

    int x = blockIdx.x * blockDim.x + threadIdx.x; // LINHA
    int y = blockIdx.y * blockDim.y + threadIdx.y; // COLUNA

    // TIRANDO A BORDA DO PROCESSAMENTO
    if ( y > lf-li || x < 2 || x > coluna-2 || (li == 0 && y < 2) || (lf==linha-1 && y > (lf-li)-2) )
        return;

    float sum = 0.0f;
    int cont = 0;

    // SE A IMAGEM NAO FOR O PRIMEIRO BLOCO
    // DEFINE O INICIO PARA DUAS LINHAS ADIANTE
    // PARA NAO PROCESSAR A BORDA
    int inicio = 0;
    if (li != 0)
        inicio = 2;

    for(int l2= -2; l2<=2; l2++) {
        for(int c2=-2; c2<=2; c2++) {
            if(l2 >= 0 && c2 >= 0) {
                sum += tex2D(textureIn, inicio+x+l2, y+c2);
                cont++;
            }
        }
    }

    // ARMAZENDO O RESULTADO
    // NA MEMORIA GLOBAL
    kOutput[y*pitch+x] = static_cast<unsigned char>(sum/cont);
}

// FUNCAO PARA APLICAR SMOOTH
// SEM TEXTURE
__global__ void kernel(unsigned char* kInput, unsigned char* kOutput,
                       const int coluna, const int linha, const int li, const int lf) {

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    int c = x % coluna; // COLUNA
    int l = (x-c)/coluna; // LINHA

    // TIRANDO A BORDA DO PROCESSAMENTO
    if ( l > lf-li || c < 2 || c > coluna-2 || (li == 0 && l < 2) || (lf==linha-1 && l > (lf-li)-2) )
        return;

    // APLICANDO O SMOOTH
    float sum = 0.0f;

    for(int l2 = -2; l2 <= 2; ++l2) {
        for(int c2 = -2; c2 <= 2; ++c2) {
            if((c+l2) >= 2 && (c+l2) < coluna-2 && (l+c2) >= -2 && (l+c2) <= lf-li+4) {
                int p = (x + 2*coluna)+(l2*coluna)+c2; // NAO E O PRIMEIRO BLOCO
                if (li == 0)
                    p = x + 2*coluna; // PRIMEIRO BLOCO
                sum += kInput[p];
            }
        }
    }

    // ARMAZENDO O RESULTADO
    // NA MEMORIA GLOBAL
    kOutput[x] = sum/25;
}

// FUNCAO PARA TRANSFORMAR A IMAGEM
// LIDA EM UM ARRAY
// NECESSARIO PARA UTILIZAR MEMORIA TEXTURA
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

// FUNCAO PARA TRANSFORMAR A IMAGEM
// DE UM ARRAY PARA UM STRUCT PADRAO DO SISTEMA
// NECESSARIO PARA UTILIZAR MEMORIA TEXTURA
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

// FUNCAO __HOST__
// PARA CHAMAR O KERNEL COM TEXTURA
float applySmoothTexture(initialParams* ct, PPMImageParams* imageParams,
                       PPMThread* thread, int numThread, cudaStream_t* streamSmooth, int filtro) {

    // INICIANDO O TEMPO
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ALOCANDO VARIAVEIS PARA COPIAR
    // E RECEBER A IMAGEM
    unsigned char *cpuIn, *cpuOut, *gpuIn, *gpuOut;
    cpuIn = (unsigned char *)malloc(thread[numThread].linhasIn * imageParams->coluna * sizeof(unsigned char) );
    cpuOut = (unsigned char *)malloc(thread[numThread].linhasOut * imageParams->coluna * sizeof(unsigned char) );

    // CONVERTENDO O PADRAO DO SISTEMA
    // PARA ARRAY
    structToArray(imageParams, thread, numThread, cpuIn, filtro);

    // ALOCANDO VARIAVEIS PARA
    // ENVIAR E RECEBER A IMAGEM
    // PARA O KERNEL
    size_t pitch = 0;
    gpuErrchk( cudaMallocPitch<unsigned char>(&gpuIn,&pitch,imageParams->coluna,thread[numThread].linhasIn) );
    gpuErrchk( cudaMallocPitch<unsigned char>(&gpuOut,&pitch,imageParams->coluna,thread[numThread].linhasOut) );

    // COPIANDO DADOS DO HOST
    // PARA O DEVICE
    gpuErrchk( cudaMemcpy2DAsync(gpuIn,pitch,cpuIn,imageParams->coluna,imageParams->coluna,thread[numThread].linhasIn,cudaMemcpyHostToDevice, streamSmooth[numThread]) );

    // ALOCANDO A IMAGEM NA
    // MEMORIA TEXTURA
    gpuErrchk( cudaBindTexture2D(NULL,textureIn,gpuIn,imageParams->coluna,thread[numThread].linhasIn,pitch) );

    // DEFININDO O BLOCO
    dim3 blockDims(16,16);
    dim3 gridDims;
    gridDims.x = (imageParams->coluna + blockDims.x - 1)/blockDims.x;
    gridDims.y = (thread[numThread].linhasIn + blockDims.y - 1)/blockDims.y;

    // CHAMANDO O KERNEL
    cudaEventRecord(start, 0); // INICIANDO O RELOGIO

    if (ct->debug >= 1)
        printf("Kernel Smooth[%d][%s] - Grid:%d, Block:%d, li:%d, lf:%d\n",
               numThread, imageParams->tipo, gridDims.x, blockDims.x, thread[numThread].li, thread[numThread].lf);

    // CHAMANDO O KERNEL
    kernelTexture<<<gridDims,blockDims, 0, streamSmooth[numThread]>>>(gpuOut,imageParams->coluna,imageParams->linha,pitch,thread[numThread].lf,thread[numThread].li);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0); // PARANDO O RELOGIO
    cudaEventSynchronize(stop);

    // COPIANDO OS DADOS
    // DO DEVICE PARA O HOST
    cudaMemcpy2DAsync(cpuOut,imageParams->coluna,gpuOut,pitch,imageParams->coluna,thread[numThread].linhasOut,cudaMemcpyDeviceToHost, streamSmooth[numThread]);

    // CONVERTENDO O ARRAY RECEBIDO
    // PARA A STRUCT PADRAO DO SISTEMA
    arrayToStruct(imageParams, thread, numThread, cpuOut, filtro);

    // LIBERANDO A TEXTURA
    cudaUnbindTexture(textureIn);

    // LIBERANDO MEMORIA
    cudaFree(gpuIn);
    cudaFree(gpuOut);
    free(cpuIn);
    free(cpuOut);

    // REGISTRANDO O TEMPO
    cudaEventElapsedTime(&time, start, stop);

    if (ct->debug >= 1)
        printf("Done Smooth[%d][%s] - linhas: %d, li:%d, lf:%d\n",
               numThread, imageParams->tipo, thread[numThread].linhasIn, thread[numThread].li, thread[numThread].lf);

    return time;
}

// FUNCAO __HOST__
// DEFINICAO DOS PARAMETROS DE CHAMADA DO KERNEL
float applySmooth(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread,
                 int numThread, cudaStream_t* streamSmooth, int filtro) {

    // INICIANDO O TEMPO
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ALOCANDO VARIAVEIS PARA COPIAR
    // E RECEBER A IMAGEM
    unsigned char *cpuIn, *cpuOut, *gpuIn, *gpuOut;
    cpuIn = (unsigned char *)malloc(thread[numThread].linhasIn * imageParams->coluna * sizeof(unsigned char) );
    cpuOut = (unsigned char *)malloc(thread[numThread].linhasOut * imageParams->coluna * sizeof(unsigned char) );

    // CONVERTENDO O PADRAO DO SISTEMA
    // PARA ARRAY
    structToArray(imageParams, thread, numThread, cpuIn, filtro);

    // ALOCANDO VARIAVEIS PARA
    // ENVIAR E RECEBER A IMAGEM
    // PARA O KERNEL
    cudaMalloc( (void**) &gpuIn, thread[numThread].linhasIn * imageParams->coluna);
    cudaMalloc( (void**) &gpuOut, thread[numThread].linhasOut * imageParams->coluna);

    // DEFINICAO DO TAMANHO PADRAO
    // DO BLOCO
    dim3 blockDims(512,1,1);
    dim3 gridDims((unsigned int) ceil((double)(thread[numThread].linhasIn * imageParams->coluna/blockDims.x)), 1, 1 );

    // COPIANDO DADOS DO HOST
    // PARA O DEVICE
    cudaMemcpyAsync( gpuIn, cpuIn, thread[numThread].linhasIn * imageParams->coluna, cudaMemcpyHostToDevice, streamSmooth[numThread] );

    cudaEventRecord(start, 0); // INICIANDO O RELOGIO

    if (ct->debug >= 1)
        printf("Kernel Smooth[%d][%s] - Grid:%d, Block:%d, li:%d, lf:%d\n",
               numThread, imageParams->tipo, gridDims.x, blockDims.x, thread[numThread].li, thread[numThread].lf);

    // CHAMANDO O KERNEL
    kernel<<<gridDims, blockDims, 0, streamSmooth[numThread]>>>(gpuIn, gpuOut, imageParams->coluna, imageParams->linha, thread[numThread].li, thread[numThread].lf);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cudaEventRecord(stop, 0); // PARANDO O RELOGIO
    cudaEventSynchronize(stop);

    // COPIANDO OS DADOS
    // DO DEVICE PARA O HOST
    cudaMemcpyAsync(cpuOut, gpuOut, thread[numThread].linhasOut * imageParams->coluna, cudaMemcpyDeviceToHost, streamSmooth[numThread] );

    // CONVERTENDO O ARRAY RECEBIDO
    // PARA A STRUCT PADRAO DO SISTEMA
    arrayToStruct(imageParams, thread, numThread, cpuOut, filtro);

    // LIBERANDO A MEMORIA
    cudaFree(gpuIn);
    cudaFree(gpuOut);
    free(cpuIn);
    free(cpuOut);

    cudaDeviceSynchronize();

    // REGISTRANDO O TEMPO
    cudaEventElapsedTime(&time, start, stop);

    if (ct->debug >= 1)
        printf("Done Smooth[%d][%s] - li:%d, lf:%d\n",
               numThread, imageParams->tipo, thread[numThread].li, thread[numThread].lf);

    return time;
}


