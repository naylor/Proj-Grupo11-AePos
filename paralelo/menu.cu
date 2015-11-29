#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "menu.cuh"

void menu(initialParams* ct, int argc, char **argv){

    files* f = listDir(ct->DIRIMGIN);

    if (argv[1]) {
        if (strcmp(argv[1], "--help") == 0) {
            printf("\tDecomposicao de imagens com Smooth\n\n");
            printf("\tNaylor Garcia Bachiega (NUSP 5567669)\n\n");

            printf("Usar: ./PPMcuda -i [IMAGEM] -a [MEMORIA ASSINCRONA] -s [MEMORIA COMPARTILHADA] -c [CARGA TRABALHO(Opcional)] -d [NIVEL DEBUG(Opcional)]\n\n");
            printf("[IMAGEM]: colocar apenas o nome do arquivo (ex. model.ppm, omitir o diretorio).\n");
            printf("[MEMORIA ASSINCRONA]: se ativado, cudaMemcpyAsync e utilizado para copia da imagem.\n");
            printf("[MEMORIA COMPARTILHADA]: se ativado, a Shared Memory e utilizada em blocos de 32x32.\n");
            printf("[CARGA TRABALHO]: numero maximo de linhas, que o Rank0 alocara para cada processo, se omitido, sera uma divisao igualitaria.\n");
            printf("[NIVEL DEBUG]: permite monitorar os eventos do sistema.\n");
            printf("\nExemplo: ./PPMcuda -i model.ppm -a 1 -s 1 -c 300 -d 1\n\n");

            // SE FOI SOLICITADO O HELP
            // FINALIZADO OS NODES...
            ct->erro = -101;
            return;
        }

        getCommandLineOptions(ct, f, argc, argv);
    }

    free(f);
}
