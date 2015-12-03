#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../common/imagem.cuh"
#include "../common/timer.cuh"
#include "../common/funcao.cuh"
#include "../cuda/host.cuh"
#include "main.cuh"


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
    int r = 11000000/imageParams->coluna;
    numMaxLinhas = r;

    // DEFINA A CARGA MAXIMA DELINHAS
    if (ct->numMaxLinhas > 0)
        numMaxLinhas = ct->numMaxLinhas;

    //if (numMaxLinhas > r) {
    //    printf("\nCarga de trabalho nao permitido. Maximo para essa imagem: %d\n", r);
    //    exit(0);
    //}

    int blocks = (imageParams->linha/numMaxLinhas)+1;

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
    cudaStream_t streamSmooth[blocks];

    if (ct->async == 1)
        for (int i = 0; i < blocks; ++i)
            cudaStreamCreate(&streamSmooth[i]);

    // ALOCA MEMORIA PARA A QUANTIDADE
    // DE BLOCOS QUE SERAO GERADOS
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

    for(int t=0; t<i; t++) {
        // FAZ A LEITURA DA PARTE DA IMAGEM
        // NO DISCO
        start_timer(tempoR); //INICIA O RELOGIO
        getImageBlocks(ct, imageParams, block,  t);
        stop_timer(tempoR);

        //applySmooth(ct, imageParams, block, t, streamSmooth);
        tempoF->timeval_diff += box_filter_8u_c1(ct, imageParams, block, t, streamSmooth);
        printf ("Time for the kernel: %f ms\n", tempoF->timeval_diff);

        // FAZ A GRAVACAO
        start_timer(tempoW); //INICIA O RELOGIO
        writePPMPixels(ct, imageParams, block, t);
        stop_timer(tempoW);


    }


    //PARA O RELOGIO
    stop_timer(tempoA);

    relogio[1].tempoF = tempoF->timeval_diff;
    relogio[1].tempoR = total_timer(tempoR);
    relogio[1].tempoW = total_timer(tempoW);
    relogio[0].tempoA = total_timer(tempoA);

    show_timer(relogio, 1);

    // DESTROI O CUDA STREAM
    if (ct->async == 1)
        for (int i = 0; i < blocks; ++i)
            cudaStreamDestroy(streamSmooth[i]);

    //ESCREVE NO ARQUIVO DE LOGS
    writeFile(ct, imageParams, tempoR, tempoW, tempoA);

    // LIMPAR A MEMORIA
    cleanMemory(ct, imageParams, block);
    free(tempoR);
    free(tempoW);
    free(tempoA);

    return 0;

}
