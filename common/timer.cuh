#include <time.h>
#include <sys/time.h>
#include <stdbool.h>

#ifndef TIMER_H_
#define TIMER_H_

typedef struct {
    struct timeval timeval_start, timeval_end;
    double timeval_diff, timeval_diff_s, timeval_diff_u;
} timer;

void start_timer(timer* t);
void show_timer(timer* t, const char* tipo);
void stop_timer(timer* t);

#endif
