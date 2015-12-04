#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../common/timer.h"
#include "../common/imagem.h"

#include "sequencial.h"
#include "main.h"

int main (int argc, char *argv[]){

    // VERIFICAR SE O USUARIO
    // CONTINUA OU SAI DO PROGRAMA
    char op=NULL;
    do {

        PPMImageParams* imageParams;

        imageParams = (PPMImageParams *)malloc(sizeof(PPMImageParams));

        // CARREGA O MENU OU SETA AS OPCOES
        // CASO INSERIDAS NA LINHA DE COMANDO
        initialParams* ct = (initialParams *)calloc(1,sizeof(initialParams));

        ct->DIRIMGIN = "images_in/";
        ct->DIRIMGOUT = "images_out/";
        ct->DIRRES = "resultados/";
        ct->typeAlg = 'S'; // SEQUENCIAL
        menu(ct, argc, argv);

        sprintf((char*) &imageParams->fileOut, "%s%s", ct->DIRIMGOUT, ct->filePath);
        sprintf((char*) &imageParams->fileIn, "%s%s", ct->DIRIMGIN, ct->filePath);

        if (ct->filePath != NULL) {
            // RELOGIO
            tempo* relogio = (tempo* )malloc(sizeof(tempo)*2);

            //CARREGA O RELOGIO
            timer* tempoA = (timer *)malloc(sizeof(timer));
            timer* tempoR = (timer *)malloc(sizeof(timer));
            timer* tempoF = (timer *)malloc(sizeof(timer));
            timer* tempoW = (timer *)malloc(sizeof(timer));

            start_timer(tempoA);

            sequencial(imageParams, ct, tempoR, tempoF, tempoW);

            //PARA O RELOGIO
            stop_timer(tempoA);

            relogio[0].tempoA = total_timer(tempoA);
            relogio[1].tempoR = total_timer(tempoR);
            relogio[1].tempoF = total_timer(tempoF);
            relogio[1].tempoW = total_timer(tempoW);

            show_timer(relogio, 1);

            // ESCREVENDO OS RESULTADOS
            // NO ARQUIVO /resultados/
            writeFile(ct, imageParams, relogio);

            // LIMPANDO A MEMORIA
            cleanMemory(ct, imageParams, NULL, relogio, tempoA, tempoR, tempoF, tempoW);

        } else {
            printf("\nOpcao invalida!\n\n");
        }

        if (!argv[1]) {
            printf("\nPressione 's' para voltar ao menu ou 'n' para sair...\n");
            int ret = scanf(" %c", &op);
            if (ct->debug >= 1) printf("\nRet OP: %d", ret);
        } else {
            op = 'n';
        }

    } while (op != 'n');

    return 0;
}
