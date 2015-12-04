#ifndef PARALELO_H_INCLUDED
#define PARALELO_H_INCLUDED

#include "menu.h"

PPMImageParams* paraleloInitParams(initialParams* ct, PPMImageParams* imageParams);
int paraleloNodeWrite(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread, int numNode);
PPMThread* paraleloNodeReadAndSmooth(initialParams* ct,  PPMImageParams* imageParams,
                                     PPMNode* node, timer* tempoR, timer* tempoF, int numNode);

#endif // PARALELO_H_INCLUDED
