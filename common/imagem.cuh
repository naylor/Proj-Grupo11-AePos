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
} PPMBlock;

typedef struct {
    int linha, coluna;
    int proxLinha;
    int posIniFileIn;
    int posIniFileOut;
    char fileIn[200];
    char fileOut[200];
    char tipo[2];
} PPMImageParams;

#include "../paralelo/menu.cuh"

void getPPMParameters(initialParams* ct, PPMImageParams* imageParams);

void writePPMHeader(initialParams* ct, PPMImageParams* imageParams);

int getDivisionBlocks(initialParams* ct, PPMImageParams *imageParams,
                      PPMBlock *block, int numBlocks,
                     int numBlock, int numMaxLinhas);

int getImageBlocks(initialParams* ct, PPMImageParams* imageParams, PPMBlock* block, int numBlock);

void writePPMPixels(initialParams* ct, PPMImageParams* imageParams,
                    PPMBlock *block, int numBlock);

#define CREATOR "NGB"
#define RGB_COMPONENT_COLOR 255

#endif // IMAGEM_H_INCLUDED
