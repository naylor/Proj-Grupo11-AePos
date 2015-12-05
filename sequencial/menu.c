#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"
#include "sequencial.h"

// MENU INICIAL
// AS OPCOES DE IMAGENS SAO CARREGADAS
// AUTOMAGICAMENTE DO DIRETORIO /images_in/
void menu(initialParams* ct, int argc, char *argv[]){
    int i = 0;
    int file = NULL;

    files* f = listDir(ct->DIRIMGIN);

    printf("\tDecomposicao de imagens com Smooth\n\n");
    printf("\tNaylor Garcia Bachiega (NUSP 5567669)\n\n");
    printf("\tPara utilizar a versao de linha de comando,\n\tuse: ./PPMseq --help\n\n");

    if (argv[1]) {
        if (strcmp(argv[1], "--help") == 0) {
            printf("Usar: ./PPMseq -i [IMAGEM] -d [NIVEL DEBUG]\n\n");
            printf("[IMAGEM]: colocar apenas o nome do arquivo (ex. model.ppm, omitir o diretorio).\n");
            printf("[NIVEL DEBUG]: permite monitorar os eventos do sistema.\n");
            printf("\nExemplo: ./PPMseq -i model.ppm -d 1\n\n");
            exit(0);
        }

        getCommandLineOptions(ct, f, argc, argv);
        if (ct->erro == -101)
            exit(0);

    } else {

        for(i = 0; i < f->total; i++) {
            printf("\t%d - %s\n", i, f->names[i]);
        }
        printf("\n\t%d - Sair\n", i);
        printf("\n\tEscolha uma imagem: ");
        int ret = scanf(" %d", &file);
        if (ct->debug == 1) printf("\nRet File: %d", ret);

        if (file == i) //Sair
            exit(1);

        ct->filePath = f->names[file];

    }

    free(f);
}
