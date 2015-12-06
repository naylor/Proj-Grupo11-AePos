#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../common/imagem.h"
#include "../common/timer.h"
#include "../common/funcao.h"

#include "cuda.cuh"
#include "main.h"

int main (int argc, char **argv){

    // ALOCA MEMORIA PARA OS PARAMETROS
    // DA IMAGEM
    PPMImageParams* imageParams = (PPMImageParams *)malloc(sizeof(PPMImageParams));

    // CARREGA O MENU OU SETA AS OPCOES
    // CASO INSERIDAS NA LINHA DE COMANDO
    initialParams* ct = (initialParams *)calloc(1,sizeof(initialParams));
    ct->DIRIMGIN = "images_in/";  //DIRETORIO DAS IMAGEMS
    ct->DIRIMGOUT = "images_out/"; //DIRETORIO DE SAIDA
    ct->DIRRES = "resultados/"; //GUARDAR OS LOGS
    ct->typeAlg = 'C'; //TIPO DE ALGORITMO, C: CUDA

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
    int numMaxLinGrids = (65535 / (imageParams->coluna / 512));

        printf("\nE %d %d %d", numMaxLinGrids, imageParams->coluna, (imageParams->coluna / 512) );

    int numMaxLinhas = imageParams->linha;


    if (numMaxLinhas > numMaxLinGrids)
        numMaxLinhas = numMaxLinGrids;



    ct->numThreads = 1;

    // DEFINA A CARGA MAXIMA DELINHAS
    if (ct->numMaxLinhas > 0 && ct->numMaxLinhas < numMaxLinhas)
        numMaxLinhas = ct->numMaxLinhas;
    else if (ct->numMaxLinhas > numMaxLinhas) {
        printf("\nEsse carga excede os parametros da GPU");
        exit(0);
    }

    int numNodes = (imageParams->linha/numMaxLinhas)+1;

    printf("\nCarga de Trabalho: %d %f", numMaxLinhas, numMaxLinGrids);
    printf("\nMemoria Textura: %s", ct->texture==1?"Ativado":"Desativado");
    printf("\nMemoria Assincrona: Ativado");

    tempo* relogio = (tempo* )malloc(sizeof(tempo) * 2);
    timer* tempoA = (timer* )malloc(sizeof(timer)); // RELOGIO APLICACAO
    timer* tempoR = (timer* )malloc(sizeof(timer)); // RELOGIO LEITURA
    timer* tempoW = (timer* )malloc(sizeof(timer)); // RELOGIO WRITE
    timer* tempoF = (timer* )malloc(sizeof(timer)); // RELOGIO FILTRO

    start_timer(tempoA); // INICIA O RELOGIO DA APLICACAO

    //GRAVA O CABECALHO DA
    //IMAGEM DE SAIDA
    writePPMHeader(ct, imageParams);

    // CRIA OS CUDA STREAM PARA ASYNC
    cudaStream_t streamSmooth[numNodes];

    for (int i = 0; i < numNodes; ++i)
        cudaStreamCreate(&streamSmooth[i]);

    // ALOCA MEMORIA PARA A QUANTIDADE
    // DE BLOCOS QUE SERAO GERADOS
    PPMNode* node = (PPMNode *)malloc(sizeof(PPMNode) * numNodes);

    int t,n;
    for(n=0; n<numNodes; n++) {
        // FAZ A DIVISAO DE LINHAS
        // POR BLOCOS
        int endOfNodes = getDivisionNodes(ct, imageParams, node, 1, n, numMaxLinhas);

        // ALOCA MEMORIA PARA A THREAD
        PPMThread* thread = getDivisionThreads(ct, imageParams, node, n);

        for(t=0; t<ct->numThreads; t++) {
            // FAZ A LEITURA DA PARTE DA IMAGEM
            // NO DISCO
            start_timer(tempoR); //INICIA O RELOGIO
            getImageThreads(ct, imageParams, thread,  t, n);
            stop_timer(tempoR);

            if (ct->texture == 1) {
                if (strcmp(imageParams->tipo, "P6")==0) {
                    relogio[1].tempoF += applySmoothTexture(ct, imageParams, thread, t, streamSmooth, 1);
                    relogio[1].tempoF += applySmoothTexture(ct, imageParams, thread, t, streamSmooth, 2);
                    relogio[1].tempoF += applySmoothTexture(ct, imageParams, thread, t, streamSmooth, 3);
                } else {
                    relogio[1].tempoF += applySmoothTexture(ct, imageParams, thread, t, streamSmooth, 1);
                }
            } else {
                if (strcmp(imageParams->tipo, "P6")==0) {
                    relogio[1].tempoF += applySmooth(ct, imageParams, thread, t, streamSmooth, 1);
                    relogio[1].tempoF += applySmooth(ct, imageParams, thread, t, streamSmooth, 2);
                    relogio[1].tempoF += applySmooth(ct, imageParams, thread, t, streamSmooth, 3);
                } else {
                    relogio[1].tempoF += applySmooth(ct, imageParams, thread, t, streamSmooth, 1);
                }
            }

            // FAZ A GRAVACAO
            start_timer(tempoW); //INICIA O RELOGIO
            writePPMPixels(ct, imageParams, thread, t, n);
            stop_timer(tempoW);
        }
        free(thread);
    }

    //PARA O RELOGIO
    stop_timer(tempoA);

    relogio[1].tempoR = total_timer(tempoR);
    relogio[1].tempoW = total_timer(tempoW);
    relogio[0].tempoA = total_timer(tempoA);

    show_timer(relogio, 1);

    // DESTROI O CUDA STREAM
    for (int i = 0; i < numNodes; ++i)
        cudaStreamDestroy(streamSmooth[i]);

    //ESCREVE NO ARQUIVO DE LOGS
    writeFile(ct, imageParams, relogio);

    // LIMPAR A MEMORIA
    cleanMemory(ct, imageParams, node, relogio, tempoA, tempoR, tempoF, tempoW);

    return 0;

}
