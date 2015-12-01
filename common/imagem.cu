#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include "imagem.cuh"

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

int getDivisionBlocks(initialParams* ct, PPMImageParams *imageParams, PPMBlock *block, int numBlocks,
                     int numBlock, int numMaxLinhas) {

    int b = 0;
    numMaxLinhas--;

    // DIVISAO POR LINHA
    // NAO E POSSIVEL POR COLUNA, GERA PROBLEMAS NA GRAVACAO
    // COM FREAD
    for(imageParams->proxLinha=imageParams->proxLinha;imageParams->proxLinha<imageParams->linha;imageParams->proxLinha+=numMaxLinhas+1){
            // SE O NUMERO DE NODES FOI ATINGIDO
            // RETORNA COM AS LINHAS PARA O NODE
            // COMECAR O PROCESSAMENTO
            if (b==numBlocks) return b;

            // PRIMEIRA LINHA DA THREAD
            block[numBlock].li = imageParams->proxLinha;

            // SE AS PROXIMAS LINHAS DA IMAGEM FOREM PEQUENAS
            // PARA FORMAR UM BLOCO 5X5, ADICIONAR ESSAS LINHAS
            // NA DIVISAO ATUAL
            if (imageParams->proxLinha+numMaxLinhas+7 >= imageParams->linha) {
                block[numBlock].lf = imageParams->linha-1;
                imageParams->proxLinha=imageParams->linha;
            } else
                block[numBlock].lf = imageParams->proxLinha+numMaxLinhas;

            if (ct->debug >= 1)
                printf("Division Block[%d], li:%d, lf:%d\n", numBlock,
                   block[numBlock].li, block[numBlock].lf);
            b++;
        }

    // RETORNA O NUMERO DE NODES
    // QUE RECEBERAM TRABALHO
    return b;
}

// ESSA FUNCAO LE O ARQUIVO DE ENTRADA
// CADA THREAD VAI LER SOMENTE AS LINHAS
// QUE SAO RESPONSAVEIS PELO PROCESSAMENTO
// TEM QUE LER SEQUENCIAL
// MAS NÂO PRECISA SER TUDO DE UMA SO VEZ
int getImageBlocks(initialParams* ct, PPMImageParams* imageParams, PPMBlock* block, int numBlock)
{

    int linhas = (block[numBlock].lf-block[numBlock].li)+1;

    // ALOCA MEMORIA PARA A IMAGEM DE SAIDA
    if (strcmp(imageParams->tipo, "P6")==0)
        block[numBlock].ppmOut = (PPMPixel *)malloc(imageParams->coluna * linhas * sizeof(PPMPixel));
    else
        block[numBlock].pgmOut = (PGMPixel *)malloc(imageParams->coluna * linhas * sizeof(PPMPixel));

    block[numBlock].linhasOut = imageParams->coluna * linhas * sizeof(PPMPixel);

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
    if (block[numBlock].li == 0 && block[numBlock].lf == imageParams->linha-1)
        offset = 0;

    // SE FOR A PRIMEIRA LINHA E NAO FOR A ULTIMA
    // LE A PARTIR DO OFFSET 0 + O SEU BLOCO + 2 LINHAS POSTERIORES
    // POIS A THREAD PRECISARA DE 2 LINHAS POSTERIORES, FORA O SEU BLOCO,
    // PARA O SMOOTH
    if (block[numBlock].li == 0 && block[numBlock].lf != 0 && block[numBlock].lf != imageParams->linha-1) {
        offset = 0;
        linhas += 2;
    }

    // SE A THREAD PEGAR UM BLOCO DO MEIO DA IMAGEM
    // LE AS DUAS ANTERIORES, AS DUAS POSTERIORES + O SEU BLOCO
    if (block[numBlock].li != 0 && block[numBlock].lf != imageParams->linha-1) {
        if (strcmp(imageParams->tipo, "P6")==0)
            offset = ((block[numBlock].li-2)*imageParams->coluna)*sizeof(PPMPixel);
        else
            offset = ((block[numBlock].li-2)*imageParams->coluna)*sizeof(PGMPixel);

        linhas += 4;
    }

    // FINAL DE ARQUIVO, LE SOMENTE AS DUAS ANTERIORES + SEU BLOCO
    if (block[numBlock].li != 0 && block[numBlock].lf == imageParams->linha-1) {
        if (strcmp(imageParams->tipo, "P6")==0)
            offset = ((block[numBlock].li-2)*imageParams->coluna)*sizeof(PPMPixel);
        else
            offset = ((block[numBlock].li-2)*imageParams->coluna)*sizeof(PGMPixel);

        linhas += 2;
    }

    if (strcmp(imageParams->tipo, "P6")==0)
        block[numBlock].ppmIn = (PPMPixel *)malloc(imageParams->coluna * linhas * sizeof(PPMPixel));
    else
        block[numBlock].pgmIn = (PGMPixel *)malloc(imageParams->coluna * linhas * sizeof(PPMPixel));


    block[numBlock].linhasIn = imageParams->coluna * linhas * sizeof(PPMPixel);

    // SETA O PONTEIRO NO ARQUIVO + O CABECALHO
    // PARA A LEITURA DE CADA THREAD
    fseek(fp, imageParams->posIniFileIn+offset, SEEK_SET);

    if (ct->debug >= 1)
        printf("Read Block[%d] posIniFileIn %d, Offset %d L[%d][%d]\n\n", imageParams->coluna * linhas * sizeof(PPMPixel),
               imageParams->posIniFileIn, offset,
               block[numBlock].li,
               block[numBlock].lf);

    // LE O ARQUIVO
    int ret;
    if (strcmp(imageParams->tipo, "P6")==0)
        ret = fread_unlocked(block[numBlock].ppmIn, 3*imageParams->coluna, linhas, fp);
    else
        ret = fread_unlocked(block[numBlock].pgmIn, imageParams->coluna, linhas, fp);

    if (ret == 0) {
        printf("Error Read Block[%d] posIniFileIn %d, Offset %d L[%d][%d]\n\n", numBlock,
               imageParams->posIniFileIn, offset,
               block[numBlock].li,
               block[numBlock].lf);
        ct->erro = -101;
        return 0;
    }

    fclose(fp);

    return 1;
}

// ESSA FUNCAO ESCREVE NO ARQUIVO
// O RESULTADO DO SMOOTH
void writePPMPixels(initialParams* ct, PPMImageParams *imageParams, PPMBlock* block, int numBlock)
{

    int linhas = (block[numBlock].lf-block[numBlock].li)+1;

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
    if (block[numBlock].li == 0)
        offset = 0;

    // PARA ESCRITA EM ALGUMA POSICAO DO ARQUIVO
    if (block[numBlock].li != 0) {
        if (strcmp(imageParams->tipo, "P6")==0)
            offset = (block[numBlock].li*imageParams->coluna)*sizeof(PPMPixel);
        else
            offset = (block[numBlock].li*imageParams->coluna)*sizeof(PGMPixel);
    }

    // SETA O PONTEIRO NO ARQUIVO
    // + O CABECALHO
    fseek(fp, imageParams->posIniFileOut+offset, SEEK_SET);

    if (ct->debug >= 1)
        printf("Write Block[%d] posIniFileIn %d, Offset %d L[%d][%d]\n\n", numBlock,
               imageParams->posIniFileOut, offset,
               block[numBlock].li,
               block[numBlock].lf);

    // GRAVA O ARQUIVO
    int ret;
    if (strcmp(imageParams->tipo, "P6")==0)
        ret = fwrite_unlocked(block[numBlock].ppmOut, 3*imageParams->coluna, linhas, fp);
    else
        ret = fwrite_unlocked(block[numBlock].pgmOut, imageParams->coluna, linhas, fp);

    if (ret == 0) {
        printf("Error Write Block[%d] posIniFileIn %d, Offset %d L[%d][%d]\n\n", numBlock,
               imageParams->posIniFileOut, offset,
               block[numBlock].li,
               block[numBlock].lf);
        ct->erro = -101;
        return;
    }
    fclose(fp);
}
