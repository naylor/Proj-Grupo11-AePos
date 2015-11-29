#include <stdlib.h>
#include <stdio.h>

#include "timer.cuh"

void start_timer(timer* t) {
    gettimeofday(&t->timeval_start, NULL);
    #ifdef __linux__
    #endif
    t->clock_start = clock();
}

void stop_timer(timer* t) {
	gettimeofday(&t->timeval_end, NULL);
	#ifdef __linux__
	#endif
	t->clock_end = clock();

	//timeval diff
	t->timeval_diff += (t->timeval_end.tv_sec - t->timeval_start.tv_sec) * 1000.0; // sec to ms
	t->timeval_diff += (t->timeval_end.tv_usec - t->timeval_start.tv_usec) / 1000.0; // us to ms
	#ifdef __linux__
	#else
		t->timespec_diff = 0;
	#endif
	//clock diff
	t->clock_diff += t->clock_end - t->clock_start;
	t->clock_diff_time += ((float) t->clock_diff / CLOCKS_PER_SEC * 1000.0);
}

void show_timer(timer* t) {
    printf("[timeval] %.2fms\n", t->timeval_diff);
    #ifdef __linux__
    #endif
    printf("[clock] %d ticks -> %.2fms\n", (int) t->clock_diff, t->clock_diff_time);
}
