#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../cuda/host.cuh"
#include "../common/imagem.cuh"
#include "../common/funcao.cuh"
#include "../common/timer.cuh"

#include "menu.cuh"

int main(int argc, char** argv) {

    // ALOCA MEMORIA PARA OS PARAMETROS
    // DA IMAGEM
    PPMImageParams* imageParams = (PPMImageParams *)malloc(sizeof(PPMImageParams));

    // CARREGA O MENU OU SETA AS OPCOES
    // CASO INSERIDAS NA LINHA DE COMANDO
    initialParams* ct = (initialParams *)calloc(1,sizeof(initialParams));
    ct->DIRIMGIN = "images_in/";  //DIRETORIO DAS IMAGEMS
    ct->DIRIMGOUT = "images_out/"; //DIRETORIO DE SAIDA
    ct->DIRRES = "resultados/"; //GUARDAR OS LOGS
    ct->typeAlg = 'C'; //TIPO DE ALGORITMO, P: PARALELO

    // CARREGA AS OPCOES DO USUARIO
    menu(ct, argc, argv);

    // SETANDO O ARQUIVO DE SAIDA E ENTRADA
    sprintf((char*) &imageParams->fileOut, "%s%s", ct->DIRIMGOUT, ct->filePath);
    sprintf((char*) &imageParams->fileIn, "%s%s", ct->DIRIMGIN, ct->filePath);

    getPPMParameters(ct, imageParams);

    //INFO DO PROCESSO ESCOLHIDO:
    printf("\n\nFile PPM %s\ncoluna: %d\nlinha: %d\nTipo: %s\n", imageParams->fileIn,
                      imageParams->coluna,
                      imageParams->linha,
                      strcmp(imageParams->tipo, "P6")==0?"COLOR":"GRAYSCALE");

    // DEFINE A QUANTIDADE DE LINHAS
    // DA IMAGEM PARA LEITURA E SMOOTH
    int numMaxLinhas = imageParams->linha;

    // SE FOI DEFINIDA A QUANTIDADE DE LINHAS
    // PELO MENU, ALTERAR AQUI
    if (ct->numMaxLinhas > 0)
        numMaxLinhas = ct->numMaxLinhas;
    else {
        int r = 14000000/imageParams->coluna;
        numMaxLinhas = r;
    }

    printf("\nCarga de Trabalho: %d", numMaxLinhas);
    printf("\nMemoria Compartilhada: %s", ct->sharedMemory==1?"Ativado":"Desativado");
    printf("\nMemoria Assincrona: %s\n", ct->async==1?"Ativado":"Desativado");

    timer* tempoC = (timer* )malloc(sizeof(timer)); // RELOGIO APLICACAO
    timer* tempoR = (timer* )malloc(sizeof(timer)); // RELOGIO LEITURA
    timer* tempoS = (timer* )malloc(sizeof(timer)); // RELOGIO SMOOTH
    timer* tempoM = (timer* )malloc(sizeof(timer)); // RELOGIO MemCpy
    timer* tempoW = (timer* )malloc(sizeof(timer)); // RELOGIO WRITE

    start_timer(tempoC); // INICIA O RELOGIO DA APLICACAO

    //GRAVA O CABECALHO DA
    //IMAGEM DE SAIDA
    writePPMHeader(ct, imageParams);

    // CRIA OS CUDA STREAM PARA ASYNC
    cudaStream_t streamSmooth[numMaxLinhas];

    if (ct->async == 1)
        for (int i = 0; i < numMaxLinhas; ++i)
            cudaStreamCreate(&streamSmooth[i]);

    // ALOCA MEMORIA PARA A QUANTIDADE
    // DE BLOCOS QUE SERAO GERADOS
    int blocks = (imageParams->linha/numMaxLinhas)+1;
    PPMBlock* block = (PPMBlock *)malloc(sizeof(PPMBlock) * blocks);

    // FAZ A DIVISAO DE LINHAS
    // POR BLOCOS
    int i=0;
    while (blocks != 0) {
        blocks = getDivisionBlocks(ct, imageParams, block, 1, i, numMaxLinhas);
        if (blocks == 0)
            continue;
        i++;
    }

    // CRIA UM THREAD PARA CADA DIVISAO
    #pragma omp parallel num_threads(i) shared(i, ct, imageParams, block, t, streamSmooth)
    {
        #pragma omp for
        for(int t=0; t<i; t++) {
            // FAZ A LEITURA DA PARTE DA IMAGEM
            // NO DISCO
            start_timer(tempoR); //INICIA O RELOGIO
            getImageBlocks(ct, imageParams, block,  t);
            stop_timer(tempoR);

            applySmooth(ct, imageParams, block, t, streamSmooth, tempoS, tempoM);

            // FAZ A GRAVACAO
            start_timer(tempoW); //INICIA O RELOGIO
            writePPMPixels(ct, imageParams, block, t);
            stop_timer(tempoW);
        }
        #pragma omp barrier
    }

    //PARA O RELOGIO
    show_timer(tempoR, "READ");
    show_timer(tempoS, "SMOOTH");
    show_timer(tempoM, "MEMCPY");
    show_timer(tempoW, "WRITE");
    stop_timer(tempoC);
    show_timer(tempoC, "CUDA");

    // DESTROI O CUDA STREAM
    if (ct->async == 1)
        for (int i = 0; i < numMaxLinhas; ++i)
            cudaStreamDestroy(streamSmooth[i]);

    //ESCREVE NO ARQUIVO DE LOGS
    //writeFile(ct, imageParams, tempo);

    // LIMPAR A MEMORIA
    cleanMemory(imageParams, block, tempoC, ct);

    return 0;

}
