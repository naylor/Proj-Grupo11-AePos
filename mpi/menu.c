#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"
#include "mpi.h"
#include "../common/funcao.h"

// MENU INICIAL
// AS OPCOES DE IMAGEM SAO CARREGADAS
// AUTOMAGICAMENTE DO DIRETORIO /images_in/
void menu(initialParams* ct, int argc, char *argv[]){

    files* f = listDir(ct->DIRIMGIN);

    if (argv[1]) {
        if (strcmp(argv[1], "--help") == 0) {
            printf("\tDecomposicao de imagens com Smooth\n\n");
            printf("\tNaylor Garcia Bachiega (NUSP 5567669)\n\n");

            printf("Usar: mpiexec -n [PROCESSOS] -f [NODES] ./PPMmpi -i [IMAGEM] -t [NUMERO THREADS(Opcional)] -c [CARGA TRABALHO(Opcional)] -r [CARGA ALEATORIA(Opcional)] -l [LEITURA INDIVIDUAL(Opcional)] -d [NIVEL DEBUG(Opcional)]\n\n");
            printf("[PROCESSOS]: numero de processos que serao gerados.\n");
            printf("[IMAGEM]: colocar apenas o nome do arquivo (ex. model.ppm, omitir o diretorio).\n");
            printf("[NODES]: substituir pelo arquivo contendo os nodes: nodes\n");
            printf("[NUMERO THREADS]: numero de threads para cada node local, se omitido, sera com base no numero de nucleos.\n");
            printf("[CARGA TRABALHO]: numero maximo de linhas, que o Rank0 alocara para cada processo, se omitido, sera uma divisao igualitaria.\n");
            printf("[CARGA ALEATORIA]: se ativado, as cargas enviadas para os nodes serao aleatorias.\n");
            printf("[LEITURA INDIVIDUAL]: faz com que cada processo tenha acesso exclusivo a imagem no momento da leitura.\n");
            printf("[NIVEL DEBUG]: permite monitorar os eventos do sistema, permitido 1: nivel do node e 2: nivel da imagem.\n");
            printf("\nExemplo: ./PPMmpi -i model.ppm -t 2 -c 300 -r 1 -l 1 -d 1\n\n");

            // SE FOI SOLICITADO O HELP
            // FINALIZADO OS NODES...
            ct->erro = -101;
            return;
        }

        getCommandLineOptions(ct, f, argc, argv);
    }

    free(f);
}
