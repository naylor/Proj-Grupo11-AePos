#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <stdbool.h>

#include "timer.h"

void start_timer(timer* t) {
    gettimeofday(&t->timeval_start, NULL);
}

void stop_timer(timer* t) {
	gettimeofday(&t->timeval_end, NULL);

	//timeval diff
	t->timeval_diff_s += t->timeval_end.tv_sec - t->timeval_start.tv_sec;
	t->timeval_diff_u += t->timeval_end.tv_usec - t->timeval_start.tv_usec;
}

double total_timer(timer* t) {
	t->timeval_diff = (t->timeval_end.tv_sec - t->timeval_start.tv_sec) * 1000.0; // sec to ms
	t->timeval_diff += (t->timeval_end.tv_usec - t->timeval_start.tv_usec) / 1000.0; // us to ms
    return t->timeval_diff;
}

void show_timer(tempo* t, int numNodes) {

    double tempoR=0;
    double tempoF=0;
    double tempoW=0;
    int i;
    for(i=1; i <= numNodes; i++) {
        tempoR += t[i].tempoR;
        tempoF += t[i].tempoF;
        tempoW += t[i].tempoW;
    }

    t[0].tempoR = tempoR/numNodes;
    t[0].tempoF = tempoF/numNodes;
    t[0].tempoW = tempoW/numNodes;

	//timeval diff
    printf("[Time Read] %.2fms\n", t[0].tempoR);
    printf("[Time Filter] %.2fms\n", t[0].tempoF);
    printf("[Time Write] %.2fms\n", t[0].tempoW);
    printf("[Time App] %.2fms\n", t[0].tempoA);
}
