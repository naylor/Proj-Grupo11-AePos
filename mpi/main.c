#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>

#include "../common/timer.h"
#include "../common/imagem.h"

#include "openmpi.h"
#include "main.h"


int main (int argc, char **argv){

    // CARREGA O MENU OU SETA AS OPCOES
    // CASO INSERIDAS NA LINHA DE COMANDO
    initialParams* ct = (initialParams *)calloc(1,sizeof(initialParams));
    ct->DIRIMGIN = "images_in/";  //DIRETORIO DAS IMAGEMS
    ct->DIRIMGOUT = "images_out/"; //DIRETORIO DE SAIDA
    ct->DIRRES = "resultados/"; //GUARDAR OS LOGS
    ct->typeAlg = 'P'; //TIPO DE ALGORITMO, P: PARALELO

    //INICIANDO MPI
    MPI_Status status;
    MPI_Request requestNull;

    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    PPMImageParams* imageParams = (PPMImageParams *)malloc(sizeof(PPMImageParams));
    PPMNode* node = (PPMNode *)malloc(sizeof(PPMNode) * size+1);

    // RELOGIO PARA CADA NODE
    tempo* relogio = (tempo* )malloc(sizeof(tempo) * size+1);

    char completedIndexes='I';
    int inteiro = 2;

    if ( rank == 0 ) {
        //VERIFICANDO SE O ARQUIVO
        //EXISTE E SETANDO OPCOES DE THREAD
        //E LINHAS MAXIMAS
        menu(ct, argc, argv);

        //DEFININDO O NUMERO DE PROCESSOS
        //DEIXANDO O RANK 0 APENAS PARA CONTROLE
        ct->numProcessos = size-1;

        if (ct->erro != -101) {
            //CONFIGURACAO DA IMAGEM
            imageParams = paraleloInitParams(ct, imageParams);

            //GRAVA O CABECALHO DA
            //IMAGEM DE SAIDA
            writePPMHeader(ct, imageParams);
        }
        //SE TIVER ALGUM ERRO DE CONFIGURACAO
        //ENVIAR COMANDO PARA OS NÓS FINALIZAREM
        int i;
        if (ct->erro == -101) {
            for(i=1; i <= ct->numProcessos; i++) {
                MPI_Ssend(&ct->erro, inteiro, MPI_INT, i, 01, MPI_COMM_WORLD);
            }
            ct->numProcessos = 0;
        }
    }

    //ENVIANDO AS CONFIGURACOES BASICAS PARA
    //OS PROCESSOS TRABALHAREM
    MPI_Bcast ( &imageParams->coluna, inteiro, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &imageParams->linha, inteiro, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &imageParams->posIniFileIn, inteiro, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &imageParams->posIniFileOut, inteiro, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &imageParams->tipo, 2, MPI_CHAR, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &imageParams->fileIn, 200, MPI_CHAR, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &imageParams->fileOut, 200, MPI_CHAR, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &ct->numThreads, inteiro, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast ( &ct->debug, inteiro, MPI_INT, 0, MPI_COMM_WORLD );

    if (rank == 0) {
        if (ct->filePath != NULL) {

            //CARREGA O RELOGIO
            timer* tempoA = (timer *)malloc(sizeof(timer));
            start_timer(tempoA);

            int gravar=0;
            int ler=0;
            printf("\nProcessos iniciados: %d\n\n", ct->numProcessos);

            //O RANK 0 VAI ABRIR UMA THREAD PARA CADA PROCESSO
            //VAI INFORMAR QUAL PEDACO DA IMAGEM O NODE VAI PROCESSAR
            //VAI ESPERAR O NODE RESPONDER COM O TERMINO DO PROCESSAMENTO
            //VAI AUTORIZAR O NODE A ESCREVER NO DISCO, EVITANDO DISPUTA
            //FOI TESTADO DISPUTA EM DISCO COM FWRITE E FWRITE_UNLOCKED
            //POREM OS RESULTADOS NAO FORAM BONS, HA FALHAS DE GRAVACAO
            //POR CONCORRENCIA.
            #pragma omp parallel num_threads(ct->numProcessos) shared(gravar, ler, relogio, status, imageParams, ct)
            {
                int i;
                //ABRE UMA THREAD PARA CADA PROCESSO
                #pragma omp for
                for(i=1; i <= ct->numProcessos; i++) {
                    int tServer = omp_get_thread_num();
                    int primeiro = 0;
                    int fim = 0;
                    //SOMENTE FINALIZA QUANDO NAO TIVER MAIS LINHAS
                    //PARA SEREM PROCESSADAS
                    while (fim == 0) {
                        //ENTRA AQUI, DEPOIS DE ENVIAR TRABALHO PARA O PROCESSO
                        //AGUARDO O PROCESSO FINALIZAR E RESPONDER
                        if (primeiro == 1) {
                            //SE A LEITURA INDIVIDUAL ESTIVER ATIVADA
                            //OS NODES PRECISARAO ENTRAR NA FILA PARA LER
                            if (ct->leituraIndividual == 1) {
                                if (ct->debug >= 1) printf("Server[%d] esperando node solicitar fila de leitura: %d\n", tServer, i);
                                int check_receive = 0;
                                while (check_receive == 0) {
                                    #pragma omp critical
                                    {
                                        MPI_Iprobe(i, 11, MPI_COMM_WORLD, &check_receive, &status);
                                    }
                                }
                                if (check_receive == 1)
                                    #pragma omp critical
                                    {
                                        MPI_Recv(&completedIndexes, 1, MPI_CHAR, i, 11, MPI_COMM_WORLD, &status);
                                    }

                                if (ct->debug >= 1) printf("Server[%d] recebe mensagem do node solicitando ler: %d\n", tServer, i);
                                int lido = 0;

                                //PROCESSO TER PERMISSAO DE LER
                                while (lido == 0) {
                                    #pragma omp critical
                                    {
                                        if (ler == 0) {
                                            ler = 1;
                                            completedIndexes = 'R';
                                            if (ct->debug >= 1) printf("Server[%d] permite node ler: %d\n", tServer, i);
                                            MPI_Ssend(&completedIndexes, 1, MPI_CHAR, i, 12, MPI_COMM_WORLD);
                                            lido = 1;
                                            MPI_Recv(&completedIndexes, 1, MPI_CHAR, i, 13, MPI_COMM_WORLD, &status);
                                            if (ct->debug >= 1) printf("Server[%d] tirando node da regiao de leitura: %d\n", tServer, i);
                                            ler=0;
                                        }
                                    }
                                }
                            }

                            if (ct->debug >= 1) printf("Server[%d] esperando node aplicar smooth: %d\n", tServer, i);
                            int check_receive = 0;
                            while (check_receive == 0) {
                                #pragma omp critical
                                {
                                    MPI_Iprobe(i, 31, MPI_COMM_WORLD, &check_receive, &status);
                                }
                            }
                            if (check_receive == 1)
                                #pragma omp critical
                                {
                                    MPI_Recv(&completedIndexes, 1, MPI_CHAR, i, 31, MPI_COMM_WORLD, &status);
                                }

                            if (ct->debug >= 1) printf("Server[%d] recebe mensagem do node solicitando gravar: %d\n", tServer, i);
                            int gravado = 0;
                            //QUANDO O PROCESSO FINALIZAR
                            //TENTA GRAVAR OS DADOS NO DISCO
                            //SE CONSEGUIR ENTRAR NA REGIAO CRITICA
                            //ALTERAR O VALOR DE "GRAVAR" PARA NENHUM
                            //PROCESSO TER PERMISSAO DE GRAVACAO
                            while (gravado == 0) {
                                #pragma omp critical
                                {
                                    if (gravar == 0) {
                                        gravar = 1;
                                        completedIndexes = 'W';
                                        if (ct->debug >= 1) printf("Server[%d] permite node gravar: %d\n", tServer, i);
                                        MPI_Ssend(&completedIndexes, 1, MPI_CHAR, i, 32, MPI_COMM_WORLD);
                                        gravado = 1;
                                        MPI_Recv(&relogio[i].tempoR, 1, MPI_FLOAT, i, 33, MPI_COMM_WORLD, &status);
                                        MPI_Recv(&relogio[i].tempoF, 1, MPI_FLOAT, i, 34, MPI_COMM_WORLD, &status);
                                        MPI_Recv(&relogio[i].tempoW, 1, MPI_FLOAT, i, 35, MPI_COMM_WORLD, &status);
                                        if (ct->debug >= 1) printf("Server[%d] tirando node da regiao de gravacao: %d\n", tServer, i);
                                        gravar=0;
                                    }
                                }
                            }
                        }
                        //CHAMA A FUNCAO getDivisionNodes
                        //ELA FORNECE PARA OS PROCESSOS A DIVISAO
                        //DE TRABALHO DE CADA UM DE ACORDO COM O NUMERO
                        //MAXIMO DE LINHAS INFORMADO
                        int blocks, maxLinhasRand;
                        if (ct->cargaAleatoria == 1)
                            maxLinhasRand = rand_interval((int)ct->numMaxLinhas/2, ct->numMaxLinhas*2);
                        else
                            maxLinhasRand = ct->numMaxLinhas;

                        #pragma omp critical
                        {
                            blocks = getDivisionNodes(ct, imageParams, node, 1, i, maxLinhasRand);
                        }
                        //ENVIA O TRABALHO PARA O PROCESSO
                        if (blocks != 0) {
                            #pragma omp critical
                            {
                                MPI_Isend(&node[i].li, inteiro, MPI_INT, i, 21, MPI_COMM_WORLD, &requestNull);
                                MPI_Isend(&node[i].lf, inteiro, MPI_INT, i, 22, MPI_COMM_WORLD, &requestNull);
                            }
                            primeiro = 1;
                            if (ct->debug >= 1) printf("Server[%d] enviando trabalho(carga: %d) para o node: %d\n", tServer, maxLinhasRand, i);
                        } else {
                            //CASO ACABOU O TRABALHO
                            //FINALIZA O PROCESSO
                            //SENAO, CONTINUA COM OUTRO BLOCO
                            if (ct->debug >= 1) printf("Server[%d] informado que o node acabou o trabalho: %d\n", tServer, i);
                            node[i].li = -101;
                            #pragma omp critical
                            {
                                MPI_Isend(&node[i].li, inteiro, MPI_INT, i, 21, MPI_COMM_WORLD, &requestNull);
                            }
                            fim=1;
                        }
                    }
                    if (ct->debug >= 1) printf("Server[%d] foi finalizado: %d\n", tServer, rank);
                }
                #pragma omp barrier
            }
            printf("\n");

            //PARA O RELOGIO
            stop_timer(tempoA);

            relogio[rank].tempoA = total_timer(tempoA);

            show_timer(relogio, ct->numProcessos);

            //ESCREVE NO ARQUIVO DE LOGS
            //writeFile(imageParams, relogio, ct);

            if (ct->debug >= 1) printf("All Server finalizados: %d\n", rank);

        } else {
            printf("\nOpcao invalida!\n\n");
            exit(0);
        }

    //FIM RANK 0
    } else {
    //INICIO DOS NODES
        timer* tempoR = (timer *)malloc(sizeof(timer));
        timer* tempoF = (timer *)malloc(sizeof(timer));
        timer* tempoW = (timer *)malloc(sizeof(timer));
        char hostname[255];
        gethostname(hostname,255);
        int stop = 0;
        //LOOP SO FINALIZA QUANDO
        //NAO HA MAIS TRABALHO PARA O PROCESSO
        while (stop==0) {
            //RECEBENDO TRABALHO DO RANK 0
            //SE LI FOR -2, SIGNIFICA QUE NAO
            //HA MAIS TRABALHO A SER FEITO
            if (ct->debug >= 1) printf("Node aguardando trabalho: %d\n", rank);
            MPI_Recv(&node[rank].li, inteiro, MPI_INT, 0, 21, MPI_COMM_WORLD, &status);

            //ENTRA AQUI CASO NAO HA MAIS TRABALHO
            if (node[rank].li == -101) {
                if (ct->debug >= 1) printf("Node nao tem mais trabalho: %d - %s\n", rank, hostname);
                stop=1;
            } else {
                //SENAO, CONTINUA RECEBENDO OS DADOS
                //PARA PROCESSAMENTO DA IMAGEM
                MPI_Recv(&node[rank].lf, inteiro, MPI_INT, 0, 22, MPI_COMM_WORLD, &status);

                PPMThread* thread;

                if (ct->leituraIndividual == 1) {
                    if (ct->debug >= 1) printf("Node solicita entrada na fila de leitura: %d\n", rank);
                    MPI_Ssend(&completedIndexes, 1, MPI_CHAR, 0, 11, MPI_COMM_WORLD);

                    //AGUARDA AUTORIZACAO DO RANK 0
                    //PARA LER
                    MPI_Recv(&completedIndexes, 1, MPI_CHAR, 0, 12, MPI_COMM_WORLD, &status);
                    if (completedIndexes == 'R')
                        if (ct->debug >= 1) printf("Node tem permissao para ler: %d - %s\n", rank, hostname);
                }

                //FAZ A DIVISAO DAS LINHAS RECEBIDAS
                //PARA AS THREADS
                //EXECUTA A LEITURA DO BLOCO DA IMAGEM
                //APLICA SMOOTH
                thread = paraleloNodeReadAndSmooth(ct, imageParams, node, tempoR, tempoF, rank);

                if (ct->leituraIndividual == 1) {
                    //INFORMA O NODE QUE ACABOU
                    MPI_Ssend(&completedIndexes, 1, MPI_CHAR, 0, 13, MPI_COMM_WORLD);
                    if (ct->debug >= 1) printf("Node informando que acabou a leitura: %d - %s\n", rank, hostname);
                }

                //INFORMA O RANK 0 QUE FINALIZOU
                //E ESTA PRONTO PARA GRAVAR
                if (ct->debug >= 1) printf("Node solicita entrada na fila de gravacao: %d\n", rank);
                MPI_Ssend(&completedIndexes, 1, MPI_CHAR, 0, 31, MPI_COMM_WORLD);

                //AGUARDA AUTORIZACAO DO RANK 0
                //PARA GRAVAR
                MPI_Recv(&completedIndexes, 1, MPI_CHAR, 0, 32, MPI_COMM_WORLD, &status);
                if (completedIndexes == 'W') {
                    if (ct->debug >= 1) printf("Node tem permissao para gravar: %d - %s\n", rank, hostname);
                    //GRAVA IMAGEM PROCESSADA NO DISCO
                    start_timer(tempoW); // INICIA O RELOGIO
                    paraleloNodeWrite(ct, imageParams, thread, rank);
                    stop_timer(tempoW); // PARA O RELOGIO

                    //INFORMA O NODE QUE ACABOU
                    //E AGUARDO POR MAIS TRABALHO

                    relogio[rank].tempoR = total_timer(tempoR);
                    relogio[rank].tempoF = total_timer(tempoF);
                    relogio[rank].tempoW = total_timer(tempoW);

                    MPI_Ssend(&relogio[rank].tempoR, 1, MPI_FLOAT, 0, 33, MPI_COMM_WORLD);
                    MPI_Ssend(&relogio[rank].tempoF, 1, MPI_FLOAT, 0, 34, MPI_COMM_WORLD);
                    MPI_Ssend(&relogio[rank].tempoW, 1, MPI_FLOAT, 0, 35, MPI_COMM_WORLD);
                    if (ct->debug >= 1) printf("Node informando que acabou a gravacao: %d - %s\n", rank, hostname);
                    free(thread);
                }
            }
        }
        if (ct->debug >= 1) printf("Node finalizado: %d\n", rank);
    }
    //FIM DOS NODES

    MPI_Barrier(MPI_COMM_WORLD);

    free(node);
    free(ct);
    free(imageParams);
    free(relogio);

    MPI_Finalize();

    return 0;
}
