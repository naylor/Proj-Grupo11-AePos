#include <stdlib.h>
#include <stdio.h>

#include "timer.cuh"

void start_timer(timer* t) {
    gettimeofday(&t->timeval_start, NULL);
}

void stop_timer(timer* t) {
	gettimeofday(&t->timeval_end, NULL);

	//timeval diff
	t->timeval_diff += (t->timeval_end.tv_sec - t->timeval_start.tv_sec) * 1000.0; // sec to ms
	t->timeval_diff += (t->timeval_end.tv_usec - t->timeval_start.tv_usec) / 1000.0; // us to ms
}

void show_timer(timer* t, const char* tipo) {
    printf("[time %s] %.2fms\n", tipo, t->timeval_diff);
}
