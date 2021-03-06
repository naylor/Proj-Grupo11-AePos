#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/funcao.h"
#include "menu.h"

void menu(initialParams* ct, int argc, char **argv){

    files* f = listDir(ct->DIRIMGIN);

    if (argv[1]) {
        if (strcmp(argv[1], "--help") == 0) {
            printf("\tDecomposicao de imagens com Smooth\n\n");
            printf("\tNaylor Garcia Bachiega (NUSP 5567669)\n\n");

            printf("Usar: ./PPMcuda -i [IMAGEM] -m [MEMORIA TEXTURA] -c [CARGA TRABALHO(Opcional)] -d [NIVEL DEBUG(Opcional)]\n\n");
            printf("[IMAGEM]: colocar apenas o nome do arquivo (ex. model.ppm, omitir o diretorio).\n");
            printf("[MEMORIA TEXTURA]: se ativado, a memoria de textura e utilizada em blocos de 16x16.\n");
            printf("[CARGA TRABALHO]: numero maximo de linhas para processamento por Stream (Assincrono).\n");
            printf("[NIVEL DEBUG]: permite monitorar os eventos do sistema.\n");
            printf("\nExemplo: ./PPMcuda -i model.ppm -a 1 -m 1 -c 300 -d 1\n\n");

            // SE FOI SOLICITADO O HELP
            exit(0);
        }

        getCommandLineOptions(ct, f, argc, argv);
    }

    free(f);
}
