#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H
#include "matrixBase.h"

typedef struct {
    int objCols; //object's cols
    int objRows; //object's rows
    int wkFrameCols; // work frame's cols
    int wkFrameRows; // work frame's rows
    int trSize; // tolerance size
    int pixelDiff; // pixel difference tolerance
    float threshold; // threoshold value of algorithm
}DeteInfo;

__global__ void ObjDetectionCuda(int *obj, int *wkFrame, float *judgeRes, DeteInfo deteInfo);

void ObjDetection(ImgInfo *obj, ImgInfo *wkFrame, float *judgeRes, DeteInfo deteInfoPar);

#endif // IMAGEPROCESS_H
