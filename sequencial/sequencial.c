#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/imagem.h"
#include "sequencial.h"
#include "main.h"

void sequencial(PPMImageParams* imageParams, initialParams* ct,
                timer* tempoR, timer* tempoF, timer* tempoW) {

    //CARREGANDO CABECALHO DA IMAGEM
    //COM O TAMANHO DA IMAGEM
    getPPMParameters(ct, imageParams);

    //INFORMANDO ONDE A LEITURA
    //DA IMAGEM DEVE COMECAR
    imageParams->proxLinha=0;

    //COMO ESSE E O SEQUENCIAL
    //SEM THREADS E SEM NODES
    ct->numThreads = 1;
    int numNodes = 1;

    //INFO DO PROCESSO ESCOLHIDO:
    printf("\n\nFile PPM %s\ncoluna: %d\nlinha: %d\nTipo: %s\n\n", imageParams->fileIn,
                      imageParams->coluna,
                      imageParams->linha,
                      strcmp(imageParams->tipo, "P6")==0?"COLOR":"GRAYSCALE");

    //COMO ESSE E SEQUENCIAL
    //IRA PROCESSAR TODAS AS LINHAS
    int numMaxLinhas = imageParams->linha;

    //ALOCANDO MEMORIA PARA ESSE NODE
    PPMNode* node = (PPMNode *)malloc(sizeof(PPMNode) * numNodes);

    //ESCREVENDO O CABECALHO NO DISCO
    writePPMHeader(ct, imageParams);

    //ESSA FUNCAO RETORNA A QUANTIDADE NECESSARIA
    //DE NODES PARA EXECUTAR A DIVISAO SOLICITADA
    //DE ACORDO COM A QUANTIDADE MAX DE LINHAS
    int endOfNodes = getDivisionNodes(ct, imageParams, node, numNodes, 0, numMaxLinhas);

    int t,n;
    for(n=0; n<endOfNodes; n++) {
        // ALOCA MEMORIA PARA A THREAD
        PPMThread* thread = getDivisionThreads(ct, imageParams, node, n);

        for(t=0; t<ct->numThreads; t++) {
            // CARREGA PARTE DA IMAGEM PARA CADA THREAD
            start_timer(tempoR);
            getImageThreads(ct, imageParams, thread,  t, n);
            stop_timer(tempoR);

            // APLICA O SMOOTH NA IMAGEM PARA CADA THREAD
            start_timer(tempoF);
            applySmooth(ct, imageParams, thread, t, n);
            stop_timer(tempoF);

            // GRAVA O RESULTADO DO SMOOTH DE CADA THREAD NO ARQUIVO
            start_timer(tempoW);
            writePPMPixels(ct, imageParams, thread, t, n);
            stop_timer(tempoW);
        }
        free(thread);
    }
    free(node);
}
