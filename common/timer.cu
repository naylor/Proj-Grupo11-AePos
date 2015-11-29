#include <stdlib.h>
#include <stdio.h>

#include "timer.cuh"

void start_timer(timer* t) {
    gettimeofday(&t->timeval_start, NULL);
}

void stop_timer(timer* t) {
	gettimeofday(&t->timeval_end, NULL);

	//timeval diff
	t->timeval_diff_s += t->timeval_end.tv_sec - t->timeval_start.tv_sec;
	t->timeval_diff_u += t->timeval_end.tv_usec - t->timeval_start.tv_usec;
}

void show_timer(timer* t, const char* tipo) {
	//timeval diff
	t->timeval_diff = t->timeval_diff_s * 1000.0; // sec to ms
	t->timeval_diff = t->timeval_diff_u / 1000.0; // us to ms

    printf("[time %s] %.2fms\n", tipo, t->timeval_diff);
}
