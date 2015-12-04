#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/imagem.h"

#include <mpi.h>
#include "main.h"

#include <omp.h>

PPMImageParams* paraleloInitParams(initialParams* ct, PPMImageParams* imageParams) {

    // SETANDO O ARQUIVO DE SAIDA E ENTRADA
    sprintf((char*) &imageParams->fileOut, "%s%s", ct->DIRIMGOUT, ct->filePath);
    sprintf((char*) &imageParams->fileIn, "%s%s", ct->DIRIMGIN, ct->filePath);

    // CARREGANDO OS VALORES INICIAIS
    // DO CABECALHO DA IMAGEM
    getPPMParameters(ct, imageParams);

    imageParams->proxLinha=0;

    // SETANDO O NUMERO DE THREADS
    // CASO NAO INFORMADO
    if (ct->numThreads == 0)
        ct->numThreads = 3;

    // SETANDO O NUMERO MAXIMO DE LINHAS
    // QUE SERAO ENVIADAS PARA CADA NODE
    // CASO NAO TENHA SIDO INFORMADO
    if (ct->numMaxLinhas == 0)
        ct->numMaxLinhas = (int)imageParams->linha/ct->numProcessos;

    if ( ct->numMaxLinhas/ct->numThreads < 20 ) {
        printf("\nAtencao, nao e possivel dividir a imagem em blocos 5x5 "
               " com o numero de processos/threads informados.\n");
        ct->erro = -101;
    }

    //INFO DO PROCESSO ESCOLHIDO:
    printf("\n\nFile PPM %s\ncoluna: %d\nlinha: %d\nTipo: %s\n\n", imageParams->fileIn,
                      imageParams->coluna,
                      imageParams->linha,
                      strcmp(imageParams->tipo, "P6")==0?"COLOR":"GRAYSCALE");
    printf("\nCarga de Trabalho: %d\nCarga Aleatoria: %d\nThreads: %d\nLeitura Individual: %d\n",
           ct->numMaxLinhas,
           ct->cargaAleatoria,
           ct->numThreads,
           ct->leituraIndividual);

    return imageParams;
}

PPMThread* paraleloNodeReadAndSmooth(initialParams* ct, PPMImageParams* imageParams,
                                     PPMNode* node, timer* tempoR, timer* tempoF, int numNode) {

    PPMThread* thread;
    // FAZ A DIVISAO DAS LINHAS RECEBIDAS
    // PARA AS THREADS
    thread = getDivisionThreads(ct, imageParams, node, numNode);
    int t;

    // ESSA PARTE FOI PARALELIZADA
    // CADA THREAD APLICA O SMOOTH
    #pragma omp parallel num_threads(ct->numThreads) shared(t, ct, imageParams, tempoR, tempoF, thread, numNode)
    {
        #pragma omp for
        for(t=0; t<ct->numThreads; t++) {
            // LEITURA DAS LINHAS DEFINIDAS
            // PARA CADA THREAD
            #pragma omp critical
            {
                start_timer(tempoR); // INICIA O RELOGIO
            }
            getImageThreads(ct, imageParams, thread,  t, numNode);
            #pragma omp critical
            {
                stop_timer(tempoR); // PARA O RELOGIO
            }
            #pragma omp critical
            {
                start_timer(tempoF);
            }
            applySmooth(ct, imageParams, thread, t, numNode); // APLICA O SMOOTH PARA CADA THREAD
            #pragma omp critical
            {
                stop_timer(tempoF);
            }
        }
        #pragma omp barrier
    }
    return thread;
}

// ESSA FUNCAO ESCREVE NO ARQUIVO DE SAIDA
// A IMAGEM COM O FILTRO APLICADO
int paraleloNodeWrite(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread, int numNode) {

    int t;

    // PARA CADA THREAD, ESCREVE A IMAGEM
    // NAO DA PARA PARELIZAR ESCRITA EM DISCO
    for(t=0; t<ct->numThreads; t++)
        writePPMPixels(ct, imageParams, thread, t, numNode);

    return 1;
}
