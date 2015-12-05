#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "imagem.h"
#include <unistd.h>
#include "funcao.h"

// ESSA FUNCAO FAZ A LEITURA DO CABECALHO
// DA IMAGEM, RETORNANDO O TAMANHO E ONDE
// O OFFSET DEVE SER SETADO PARA LEITURA
// E GRAVACAO
void getPPMParameters(initialParams* ct, PPMImageParams *imageParams) {

    FILE *fp;
    char buff[16];
    int co, rgb_comp_color;

    fp = fopen(imageParams->fileIn, "rb");
    if (!fp) {
        fprintf(stderr, "Nao foi possivel abrir o arquivo: '%s'\n", imageParams->fileIn);
        ct->erro = -101;
        return;
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(imageParams->fileIn);
        ct->erro = -101;
        return;
    }

    if (buff[0] != 'P' && (buff[1] != '6' || buff[1] != '5')) {
        fprintf(stderr, "Formato da imagem invalido (Formato correto: 'P6' ou 'P5')\n");
        ct->erro = -101;
        return;
    } else {
        sprintf((char*) &imageParams->tipo, "%c%c", buff[0], buff[1]);
    }

    if (!imageParams) {
        fprintf(stderr, "Nao foi possivel alocar memoria.\n");
        ct->erro = -101;
    }

    co = getc(fp);
    while (co == '#') {
    while (getc(fp) != '\n') ;
         co= getc(fp);
    }

    ungetc(co, fp);

    if (fscanf(fp, "%d %d", &imageParams->coluna, &imageParams->linha) != 2) {
        fprintf(stderr, "Imagem com tamanho incorreto: '%s'\n", imageParams->fileIn);
        ct->erro = -101;
        return;
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "RGB invalido: '%s'\n", imageParams->fileIn);
        ct->erro = -101;
        return;
    }

    if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
         fprintf(stderr, "'%s' nao e 8-bits.\n", imageParams->fileIn);
         ct->erro = -101;
        return;
    }

    while (fgetc(fp) != '\n') ;

    // GRAVANDO O OFFSET PARA LEITURA POSTERIOR
    fflush(fp);
    imageParams->posIniFileIn = ftell(fp);

    fclose(fp);
}

// ESSA FUNCAO GRAVA O CABECALHO DA NOVA
// IMAGEM COM SMOOTH E GUARDA O OFFSET
// PARA GRAVACAO DOS REGISTROS PELAS THREADS
void writePPMHeader(initialParams* ct, PPMImageParams *imageParams)
{
    FILE *fp;
    fp = fopen(imageParams->fileOut, "wb");

    if (!fp) {
        fprintf(stderr, "Falha ao criar cabecalho: %s\n", imageParams->fileOut);
        ct->erro = -101;
        return;
    }

    fprintf(fp, "%s\n", imageParams->tipo);
    fprintf(fp, "# Created by %s\n",CREATOR);
    fprintf(fp, "%d %d\n",imageParams->coluna,imageParams->linha);
    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

    fflush(fp);
    // GRAVANDO O OFFSET PARA LEITURA POSTERIOR
    imageParams->posIniFileOut = ftell(fp);

    fclose(fp);
}

int getDivisionNodes(initialParams* ct, PPMImageParams *imageParams, PPMNode *node, int numNodes,
                     int numNode, int numMaxLinhas) {

    int b = 0;
    numMaxLinhas--;

    // DIVISAO POR LINHA
    // NAO E POSSIVEL POR COLUNA, GERA PROBLEMAS NA GRAVACAO
    // COM FREAD
    for(imageParams->proxLinha=imageParams->proxLinha;imageParams->proxLinha<imageParams->linha;imageParams->proxLinha+=numMaxLinhas+1){
            // SE O NUMERO DE NODES FOI ATINGIDO
            // RETORNA COM AS LINHAS PARA O NODE
            // COMECAR O PROCESSAMENTO
            if (b==numNodes) return b;

            // PRIMEIRA LINHA DA THREAD
            node[numNode].li = imageParams->proxLinha;

            // SE AS PROXIMAS LINHAS DA IMAGEM FOREM PEQUENAS
            // PARA FORMAR UM BLOCO 5X5, ADICIONAR ESSAS LINHAS
            // NA DIVISAO ATUAL
            if (imageParams->proxLinha+numMaxLinhas+7 >= imageParams->linha) {
                node[numNode].lf = imageParams->linha-1;
                imageParams->proxLinha=imageParams->linha;
            } else
                node[numNode].lf = imageParams->proxLinha+numMaxLinhas;

            if (ct->debug >= 2)
                printf("Division Node[%d], li:%d, lf:%d\n", numNode,
                   node[numNode].li, node[numNode].lf);
            b++;
        }

    // RETORNA O NUMERO DE NODES
    // QUE RECEBERAM TRABALHO
    return b;
}

// ESSA FUNCAO FAZ A DIVISAO DE THREADS
// A DIVISAO TAMBEM OCORRE POR LINHAS
// E RETORNA O PONTEIRO PARA O STRUCT
PPMThread* getDivisionThreads(initialParams* ct, PPMImageParams* imageParams, PPMNode *node, int numNode) {

    int linhas = (node[numNode].lf-node[numNode].li);

    int numElementos;
    int t=0;
    int proxLinha=node[numNode].li;

    // ALOCA MEMORIA PARA AS THREADS
    PPMThread* thread = (PPMThread *)malloc(sizeof(PPMThread) * ct->numThreads);

    // ACHA O NUMERO DE ELEMETOS QUE CADA
    // THREAD VAI RECEBER
    numElementos = (int)linhas/ct->numThreads;

    // COMECA A DIVISAO POR LINHA
    // MELHOR PARA CALCULAR 5x5 (NECESSARIO LER MENOS PIXELS)
    for(t=0; t <= ct->numThreads-1; t++) {

        thread[t].li = proxLinha;

        // SE AS PROXIMAS LINHAS DA IMAGEM FOREM PEQUENAS
        // PARA FORMAR UM BLOCO 5X5, ADICIONAR ESSAS LINHAS
        // NA DIVISAO ATUAL
        if (t == ct->numThreads-1) {
            thread[t].lf = node[numNode].lf;
        } else
            thread[t].lf = proxLinha+numElementos;

        proxLinha+=numElementos+1;

        // ALOCANDO MEMORIA PARA A IMAGEM DE ENTRADA
        if (thread[t].li == 0 && thread[t].lf != 0 && thread[t].lf != imageParams->linha-1)
            linhas += 2;
        if (thread[t].li != 0 && thread[t].lf != imageParams->linha-1)
            linhas += 4;
        if (thread[t].li != 0 && thread[t].lf == imageParams->linha-1)
            linhas += 2;

        if (strcmp(imageParams->tipo, "P6")==0)
            thread[t].ppmIn = (PPMPixel *)malloc(imageParams->coluna * (linhas+1) * sizeof(PPMPixel));
        else
            thread[t].pgmIn = (PGMPixel *)malloc(imageParams->coluna * (linhas+1) * sizeof(PPMPixel));

        if (ct->debug >= 2)
            printf("Division Thread(linha)[%d][%d], li:%d, lf:%d\n", numNode, t,
                        thread[t].li,
                        thread[t].lf);
    }
    return thread;
}

// ESSA FUNCAO LE O ARQUIVO DE ENTRADA
// CADA THREAD VAI LER SOMENTE AS LINHAS
// QUE SAO RESPONSAVEIS PELO PROCESSAMENTO
// TEM QUE LER SEQUENCIAL
// MAS NÂO PRECISA SER TUDO DE UMA SO VEZ
int getImageThreads(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread, int numThread, int numNode)
{

    int linhas = (thread[numThread].lf-thread[numThread].li)+1;

    // ALOCA MEMORIA PARA A IMAGEM DE SAIDA
    if (strcmp(imageParams->tipo, "P6")==0) {
        thread[numThread].ppmOut = (PPMPixel *)malloc(imageParams->coluna * linhas * sizeof(PPMPixel));
        thread[numThread].linhasOut = imageParams->coluna * linhas * sizeof(PPMPixel);
    } else {
        thread[numThread].pgmOut = (PGMPixel *)malloc(imageParams->coluna * linhas * sizeof(PGMPixel));
        thread[numThread].linhasOut = imageParams->coluna * linhas * sizeof(PGMPixel);
    }

    FILE *fp;
    fp = fopen(imageParams->fileIn, "rb");
    if (!fp) {
        fprintf(stderr, "Nao foi possivel abrir o arquivo: '%s'\n", imageParams->fileIn);
        ct->erro = -101;
        return 0;
    }

    // DEFININDO A POSICAO DE LEITURA NO ARQUIVO
    int offset;

    // SE FOR LINHA INICIAL 0 E A FINAL 0, DEFINE O OFFSET COMO 0
    if (thread[numThread].li == 0 && thread[numThread].lf == imageParams->linha-1)
        offset = 0;

    // SE FOR A PRIMEIRA LINHA E NAO FOR A ULTIMA
    // LE A PARTIR DO OFFSET 0 + O SEU BLOCO + 2 LINHAS POSTERIORES
    // POIS A THREAD PRECISARA DE 2 LINHAS POSTERIORES, FORA O SEU BLOCO,
    // PARA O SMOOTH
    if (thread[numThread].li == 0 && thread[numThread].lf != 0 && thread[numThread].lf != imageParams->linha-1) {
        offset = 0;
        linhas += 2;
    }

    // SE A THREAD PEGAR UM BLOCO DO MEIO DA IMAGEM
    // LE AS DUAS ANTERIORES, AS DUAS POSTERIORES + O SEU BLOCO
    if (thread[numThread].li != 0 && thread[numThread].lf != imageParams->linha-1) {
        if (strcmp(imageParams->tipo, "P6")==0)
            offset = ((thread[numThread].li-2)*imageParams->coluna)*sizeof(PPMPixel);
        else
            offset = ((thread[numThread].li-2)*imageParams->coluna)*sizeof(PGMPixel);

        linhas += 4;
    }

    // FINAL DE ARQUIVO, LE SOMENTE AS DUAS ANTERIORES + SEU BLOCO
    if (thread[numThread].li != 0 && thread[numThread].lf == imageParams->linha-1) {
        if (strcmp(imageParams->tipo, "P6")==0)
            offset = ((thread[numThread].li-2)*imageParams->coluna)*sizeof(PPMPixel);
        else
            offset = ((thread[numThread].li-2)*imageParams->coluna)*sizeof(PGMPixel);

        linhas += 2;
    }


    thread[numThread].linhasIn = imageParams->coluna * linhas;

    // SETA O PONTEIRO NO ARQUIVO + O CABECALHO
    // PARA A LEITURA DE CADA THREAD
    fseek(fp, imageParams->posIniFileIn+offset, SEEK_SET);

    // LE O ARQUIVO
    int ret;
    if (strcmp(imageParams->tipo, "P6")==0)
        ret = fread_unlocked(thread[numThread].ppmIn, 3*imageParams->coluna, linhas, fp);
    else
        ret = fread_unlocked(thread[numThread].pgmIn, imageParams->coluna, linhas, fp);

    if (ct->debug >= 2)
        printf("Read Thread[%d][%d] posIniFileIn %d, Offset %d L[%d][%d] R[%d]\n\n", numNode, numThread,
               imageParams->posIniFileIn, offset,
               thread[numThread].li,
               thread[numThread].lf, ret);

    if (ret == 0) {
        printf("Error Read Thread");
        ct->erro = -101;
        return 0;
    }

    fclose(fp);

    return 1;
}

// ESSA FUNCAO ESCREVE NO ARQUIVO
// O RESULTADO DO SMOOTH
void writePPMPixels(initialParams* ct, PPMImageParams *imageParams, PPMThread* thread, int numThread, int numNode)
{

    int linhas = (thread[numThread].lf-thread[numThread].li)+1;

    FILE *fp;

    fp = fopen(imageParams->fileOut, "r+");
    if (!fp) {
        fprintf(stderr, "Falha ao gravar os pixels: %s\n", imageParams->fileOut);
        ct->erro = -101;
        return;
    }

    // DEFININDO O OFFISET PARA
    // ESCRTIA DE CADA THREAD
    int offset;

    // PARA ESCRITA NO INICIO DO ARQUIVO
    if (thread[numThread].li == 0)
        offset = 0;

    // PARA ESCRITA EM ALGUMA POSICAO DO ARQUIVO
    if (thread[numThread].li != 0) {
        if (strcmp(imageParams->tipo, "P6")==0)
            offset = (thread[numThread].li*imageParams->coluna)*sizeof(PPMPixel);
        else
            offset = (thread[numThread].li*imageParams->coluna)*sizeof(PGMPixel);
    }

    // SETA O PONTEIRO NO ARQUIVO
    // + O CABECALHO
    fseek(fp, imageParams->posIniFileOut+offset, SEEK_SET);

    // GRAVA O ARQUIVO
    int ret;
    if (strcmp(imageParams->tipo, "P6")==0)
        ret = fwrite_unlocked(thread[numThread].ppmOut, 3*imageParams->coluna, linhas, fp);
    else
        ret = fwrite_unlocked(thread[numThread].pgmOut, imageParams->coluna, linhas, fp);

    if (ct->debug >= 2)
        printf("Write Thread[%d][%d] posIniFileIn %d, Offset %d L[%d][%d] R[%d]\n\n", numNode, numThread,
               imageParams->posIniFileOut, offset,
               thread[numThread].li,
               thread[numThread].lf, ret);

    if (ret == 0) {
        printf("Error Write Thread");
        ct->erro = -101;
        return;
    }

    fclose(fp);
}

// ESSA FUNCAO APLICA O SMOOTH
// NA IMAGEM
void applySmooth(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread, int numThread, int numNode)
{
    int c,l,c2,l2,p;

    int linhas = (thread[numThread].lf-thread[numThread].li)+1;

    if (ct->debug >= 2)
        printf("Apply Smooth[%d][%d] - C[%d] L[%d] li:%d,lf:%d\n",
               numNode, numThread, imageParams->coluna, linhas,
               thread[numThread].li, thread[numThread].lf);

    int k=0;

    // DEFININDO O INICIO DE LEITURA
    // PARA APLICACAO DO SMOOTH
    // NECESSARIO POIS FORAM LIDOS
    // BLOCOS A MAIS PARA CADA THREAD
    // DEPENDENDO DA POSICAO
    int inicio = 0;

    // SE A THREAD COMECOU NO INICIO DA IMAGEM
    // NADA ANTERIOR FOI LIDO
    if (thread[numThread].li == 0)
        inicio = 0;

    // SE A THREAD COMECOU EM ALGUM
    // LUGAR DA IMAGEM, 2 LINHAS ANTERIORES
    // FORAM LIDAS, COMECAR ENTAO A APLICACAO
    // DO SMOOTH APOS ELAS
    if (thread[numThread].li != 0)
        inicio = 2;

    // PERCORRENDO OS PIXELS DO
    // PEDACO DA IMAGEM LIDA
    for(l=inicio;l<=(linhas-1)+inicio;l++) {
        for(c=0;c<=imageParams->coluna-1;c++) {

            int sumr=0;
            int sumb=0;
            int sumg=0;

            // SELECIONANDO OS PIXELS VIZINHOS
            // PARA CADA PIXEL NA MATRIZ
            for(l2=l-2;l2<=l+2;l2++){
                for(c2=c-2;c2<=c+2;c2++){

                    // SOMA APENAS SE NAO FOR PIXEL DE BORDA
                    // SE FOR, A SOMO SERA EQUIVALENTE A ZERO
                    if (l2 >= 0 && c2 >= 0) {
                        p = (l2*imageParams->coluna)+c2;
                        if (strcmp(imageParams->tipo, "P6")==0) {
                            sumb += thread[numThread].ppmIn[p].blue;
                            sumg += thread[numThread].ppmIn[p].green;
                            sumr += thread[numThread].ppmIn[p].red;
                        }

                        if (strcmp(imageParams->tipo, "P5")==0)
                            sumg += thread[numThread].pgmIn[p].gray;
                    }
                }
            }

            // GUARDA O RESULTADO NA IMAGEM DE SAIDA
            if (strcmp(imageParams->tipo, "P6")==0) {
                thread[numThread].ppmOut[k].red = sumr/25;
                thread[numThread].ppmOut[k].green = sumg/25;
                thread[numThread].ppmOut[k].blue = sumb/25;
            }
            if (strcmp(imageParams->tipo, "P5")==0)
                thread[numThread].pgmOut[k].gray = sumg/25;

            k++;
        }
    }
    if (ct->debug >= 2)
        printf("Done Smooth[%d][%d] - K[%d] \n", numNode, numThread, k);
}
