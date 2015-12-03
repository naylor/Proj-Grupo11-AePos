#include <time.h>
#include <sys/time.h>
#include <stdbool.h>

#ifndef TIMER_H_
#define TIMER_H_

#ifdef __linux__ // || _unix_ || _unix
	typedef struct {
		struct timeval timeval_start, timeval_end;
		double timeval_diff, timeval_diff_s, timeval_diff_u;
	} timer;
#else
	typedef struct {
		struct timeval timeval_start, timeval_end;
		double timeval_diff, timeval_diff_s, timeval_diff_u;
	} timer;
#endif

typedef struct {
    float tempoA; // TEMPO DA APLICACAO
    float tempoR; // TEMPO DA LEITURA
    float tempoF; // TEMPO DA FILTRO
    float tempoW; // TEMPO DA GRAVACAO
} tempo;

void start_timer(timer* t);
void stop_timer(timer* t);
double total_timer(timer* t);
void show_timer(tempo* t, int numNodes);

#endif
