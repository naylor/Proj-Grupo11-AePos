#ifndef IMAGEM_H_INCLUDED
#define IMAGEM_H_INCLUDED

typedef struct {
    unsigned char red,green,blue;
} PPMPixel;

typedef struct {
    unsigned char gray;
} PGMPixel;

typedef struct {
    PPMPixel *ppmIn;
    PPMPixel *ppmOut;
    PGMPixel *pgmIn;
    PGMPixel *pgmOut;
    int li, lf;
    int linhasIn;
    int linhas;
    int linhasOut;
    int finalizado;
} PPMThread;

typedef struct {
    int li, lf;
} PPMNode;

typedef struct {
    int linha, coluna;
    int proxLinha;
    int posIniFileIn;
    int posIniFileOut;
    char fileIn[200];
    char fileOut[200];
    char tipo[2];
} PPMImageParams;

#include "../mpi/menu.h"

void getPPMParameters(initialParams* ct, PPMImageParams* imageParams);
void writePPMHeader(initialParams* ct, PPMImageParams* imageParams);
int getDivisionNodes(initialParams* ct, PPMImageParams* imageParams, PPMNode* node, int numNodes, int numNode, int numMaxLinhas);
PPMThread* getDivisionThreads(initialParams* ct, PPMImageParams* imageParams, PPMNode* node, int numNode);
int getImageThreads(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread, int numThread, int numNode);
void writePPMPixels(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread, int numThread, int numNode);
void applySmooth(initialParams* ct, PPMImageParams* imageParams, PPMThread* thread, int numThread, int numNode);

#define CREATOR "NGB"
#define RGB_COMPONENT_COLOR 255

#endif // IMAGEM_H_INCLUDED
