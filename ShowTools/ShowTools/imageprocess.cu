#include "imageprocess.h"

__global__ void ObjDetectionCuda(int *obj,
                                 int *wkFrame,
                                 float *judgeRes,
                                 DeteInfo deteInfo) {

    int glIndex = threadIdx.x + blockDim.x*blockIdx.x;
//    if(!wkFrame[glIndex]) {
//        printf("aacc\n");
//    }
//    printf("%d\n", glIndex);
//    printf("%d %d %d %d\n", deteInfo.objRows, deteInfo.objCols, deteInfo.wkFrameRows, deteInfo.wkFrameCols);
    /* calculate obj working position at wkFrame */
    int objPosRow = glIndex / deteInfo.wkFrameCols;
    int objPosCol = glIndex % deteInfo.wkFrameCols;
    if((objPosRow >= deteInfo.wkFrameRows)||
       (objPosCol >= deteInfo.wkFrameCols)) {
        return;
    }
    int allPoints = 0; // all object pixel number
    int reasonPointCnt = 0; // reasonable points number
    for(int row = 0;row < deteInfo.objRows;row++) {
        for(int col = 0;col < deteInfo.objCols;col++) {
            int objIndex = row * deteInfo.objCols + col;
            int wkObjFrameRow = row + objPosRow;
            int wkObjFrameCol = col + objPosCol;
            if(!obj[objIndex]) {
                allPoints++;
            } else {
                continue;
            }
            if((wkObjFrameRow >= deteInfo.wkFrameRows)||
               (wkObjFrameCol >= deteInfo.wkFrameCols)) {
                continue;
            }
            for(int moveParRow = -deteInfo.trSize;moveParRow < deteInfo.trSize;moveParRow++) {
                bool isbreak = false;
                for(int moveParCol = -deteInfo.trSize;moveParCol < deteInfo.trSize;moveParCol++) {
                    int wkObjFrameRowPar = wkObjFrameRow + moveParRow;
                    int wkObjFrameColPar = wkObjFrameCol + moveParCol;
                    if((wkObjFrameRowPar < 0)||
                       (wkObjFrameColPar < 0)||
                       (wkObjFrameRowPar >= deteInfo.wkFrameRows)||
                       (wkObjFrameColPar >= deteInfo.wkFrameCols)) {
                        continue;
                    } else {
//                        printf("pixel inside\n");
                    }
                    int wkObjFrameDeteIndex = deteInfo.wkFrameCols * wkObjFrameRowPar + wkObjFrameColPar;
                    if(!wkFrame[wkObjFrameDeteIndex]) {
                        reasonPointCnt++;
                        isbreak = true;
                        break;
                    }
                }
                if(isbreak) {
                    break;
                }
            }
        }
    }
    if(!allPoints) {
        return;
    } else {
//        printf("allpoints: %d\n", allPoints);
    }
    float deteRes = (float)(reasonPointCnt) / (float)(allPoints);
//    if(reasonPointCnt) {
//        printf("%d\n", reasonPointCnt);
//    }
//    if(deteRes) {
//        printf("%f\n", deteRes);
//    }
    judgeRes[glIndex] = deteRes;
}

void ObjDetection(ImgInfo *obj,
                  ImgInfo *wkFrame,
                  float *judgeRes,
                  DeteInfo deteInfoPar) {

    /*create obj array for cuda*/
    int *objCuda;
    cudaError_t erMark = cudaMalloc(&objCuda, sizeof(int)*obj->rows*obj->cols);
//    printf("objCuda: %s\n", cudaGetErrorString(erMark));
    cudaMemcpy(objCuda, obj->img, sizeof(int)*obj->rows*obj->cols, cudaMemcpyHostToDevice);

    /*create working frame for cuda*/
    int *wkFrameCuda;
    erMark = cudaMalloc(&wkFrameCuda, sizeof(int)*wkFrame->rows*wkFrame->cols);
//    printf("wkFrameCreateCuda: %s\n", cudaGetErrorString(erMark));
    erMark = cudaMemcpy(wkFrameCuda, wkFrame->img, sizeof(int)*wkFrame->rows*wkFrame->cols, cudaMemcpyHostToDevice);
//    printf("wkFrameCopyCuda: %s\n", cudaGetErrorString(erMark));

    /*create judge result for cuda*/
    float *judgeResCuda;
    erMark = cudaMalloc(&judgeResCuda, sizeof(float)*wkFrame->rows*wkFrame->cols);
//    printf("judgeResCuda: %s\n", cudaGetErrorString(erMark));

    /*set detection algorithm's parameter*/
    DeteInfo deteInfo;
    deteInfo.objCols = obj->cols;
    deteInfo.objRows = obj->rows;
    deteInfo.wkFrameCols = wkFrame->cols;
    deteInfo.wkFrameRows = wkFrame->rows;
    deteInfo.trSize = deteInfoPar.trSize;
    deteInfo.pixelDiff = deteInfoPar.pixelDiff;
    deteInfo.threshold = deteInfoPar.threshold;

    /*execute detection algorithm*/
    int perThread = 128;
    int blockCnt = (wkFrame->rows*wkFrame->cols) / perThread + 1;
    ObjDetectionCuda<<<blockCnt, perThread>>>(objCuda, wkFrameCuda, judgeResCuda, deteInfo);
    erMark = cudaMemcpy(judgeRes, judgeResCuda, sizeof(float)*wkFrame->rows*wkFrame->cols, cudaMemcpyDeviceToHost);
//    printf("finalResCuda: %s\n", cudaGetErrorString(erMark));
}
