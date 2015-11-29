#include <time.h>
#include <sys/time.h>
#include <stdbool.h>

#ifndef TIMER_H_
#define TIMER_H_

typedef struct {
    struct timeval timeval_start, timeval_end;
    struct timespec timespec_start, timespec_end;
    clock_t clock_start, clock_end, clock_diff;
    double timeval_diff, timespec_diff, clock_diff_time;
} timer;

void start_timer(timer* t);
void show_timer(timer* t, char* tipo);
void stop_timer(timer* t);

#endif
