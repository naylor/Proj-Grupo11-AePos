#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../common/imagem.h"
#include "../common/timer.h"
#include "../common/funcao.h"

#include "host.cuh"
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
    int numMaxLinhas = imageParams->linha;
    ct->numThreads = 1;

    // SE FOI DEFINIDA A QUANTIDADE DE LINHAS
    // PELO MENU, ALTERAR AQUI
    int r = 11000000/imageParams->coluna;
    numMaxLinhas = r;

    // DEFINA A CARGA MAXIMA DELINHAS
    if (ct->numMaxLinhas > 0)
        numMaxLinhas = ct->numMaxLinhas;

    //if (numMaxLinhas > r) {
    //    printf("\nCarga de trabalho nao permitido. Maximo para essa imagem: %d\n", r);
    //    exit(0);
    //}

    int numNodes = (imageParams->linha/numMaxLinhas)+1;

    printf("\nCarga de Trabalho: %d", numMaxLinhas);
    printf("\nMemoria Compartilhada: %s", ct->sharedMemory==1?"Ativado":"Desativado");
    printf("\nMemoria Assincrona: %s\n", ct->async==1?"Ativado":"Desativado");

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

    if (ct->async == 1)
        for (int i = 0; i < numNodes; ++i)
            cudaStreamCreate(&streamSmooth[i]);

    // ALOCA MEMORIA PARA A QUANTIDADE
    // DE BLOCOS QUE SERAO GERADOS
    PPMNode* node = (PPMNode *)malloc(sizeof(PPMNode) * numNodes);

    int t,n,cont=0;
    for(n=0; n<numNodes; n++) {
        // FAZ A DIVISAO DE LINHAS
        // POR BLOCOS
        int endOfNodes = getDivisionNodes(ct, imageParams, node, 1, n, numMaxLinhas);

        // ALOCA MEMORIA PARA A THREAD
        PPMThread* thread = getDivisionThreads(ct, imageParams, node, n);

    #pragma omp parallel num_threads(ct->numThreads) shared(c, ct, imageParams, thread, n)
    {
        #pragma omp for
        for(int c=0; c<ct->numThreads; c++) {
            while (thread[c].finalizado != 1);
            writePPMPixels(ct, imageParams, thread, c, n);
        }
    }

        for(t=0; t<ct->numThreads; t++) {
            // FAZ A LEITURA DA PARTE DA IMAGEM
            // NO DISCO
            start_timer(tempoR); //INICIA O RELOGIO
            getImageThreads(ct, imageParams, thread,  t, n);
            stop_timer(tempoR);

            //applySmooth(ct, imageParams, thread, t, streamSmooth);
            if (strcmp(imageParams->tipo, "P6")==0) {
                relogio[1].tempoF += box_filter_8u_c1(ct, imageParams, thread, t, streamSmooth, 1);
                relogio[1].tempoF += box_filter_8u_c1(ct, imageParams, thread, t, streamSmooth, 2);
                relogio[1].tempoF += box_filter_8u_c1(ct, imageParams, thread, t, streamSmooth, 3);
            } else
                relogio[1].tempoF += box_filter_8u_c1(ct, imageParams, thread, t, streamSmooth, 1);

            thread[t].finalizado = 1;
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
    if (ct->async == 1)
        for (int i = 0; i < numNodes; ++i)
            cudaStreamDestroy(streamSmooth[i]);

    //ESCREVE NO ARQUIVO DE LOGS
    writeFile(ct, imageParams, relogio);

    // LIMPAR A MEMORIA
    cleanMemory(ct, imageParams, node, relogio, tempoA, tempoR, tempoF, tempoW);

    return 0;

}
