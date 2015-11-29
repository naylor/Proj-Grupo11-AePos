#include <stdlib.h>
#include <stdio.h>

#include "timer.cuh"

void start_timer(timer* t) {
    gettimeofday(&t->timeval_start, NULL);
    t->clock_start = clock();
}

void stop_timer(timer* t) {
	gettimeofday(&t->timeval_end, NULL);
	t->clock_end = clock();

	//timeval diff
	t->timeval_diff += (t->timeval_end.tv_sec - t->timeval_start.tv_sec) * 1000.0; // sec to ms
	t->timeval_diff += (t->timeval_end.tv_usec - t->timeval_start.tv_usec) / 1000.0; // us to ms

	//clock diff
	t->clock_diff += t->clock_end - t->clock_start;
	t->clock_diff_time += ((float) t->clock_diff / CLOCKS_PER_SEC * 1000.0);
}

void show_timer(timer* t, const char* tipo) {
    printf("[time %s] %.2fms\n", tipo, t->timeval_diff);
     printf("[clock %s] %d ticks -> %.2fms\n", tipo, (int) t->clock_diff, t->clock_diff_time);
}
