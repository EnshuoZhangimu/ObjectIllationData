#ifndef MATRIXBASE_H
#define MATRIXBASE_H

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>

typedef struct {
    float *matrix; // matrix in 1d array like matlab's format
    unsigned int rows; // rows of matrix
    unsigned int cols; // cols of matrix
} Matrix;

typedef struct
{
   unsigned int row; // row of matrix coordinate
   unsigned int col; // col of matrix coordinate
   float eleValue; // value of matrix(row, col)
   unsigned int rows; // rows of matrix
   unsigned int cols; // cols of matrix
}MatrixCoor; // matrix coordinate structure

typedef struct
{
    float *LEqArray; // LEq array, LEq:A0 B0 C0 x0 y0 z0
    float *PEqArray; // PEq array, PEq:A1 B1 C1 x1 y1 z1
    float *PrjArray; // projection point array x y z
    int Cnt; // number of LEq and PEq
}GeoCalCache; // Analytic Geometry calculation input and result

typedef struct
{
    float *coorXYX; //coordinate axis' direction vector
    float *glCoor; //working point's coordinate in global coordinate
    float *resCoor; //calculation result
}CoorCvtData;

typedef struct
{
    int rows; // rows of image (height)
    int cols; // cols of image (width)
    int row; // working row
    int col; // working col
    int index; // working index b g r
    int pixelValue; // pixel value set or get from (row, col, index)
    int blockCnt; // working part blockCnt
    int perThread; // working part perThread
    int startIndex; // working part start index (object's cloud point)
    int *img; // img data array
    float *coorRes; // coordinate data from file
    float *cloud; //point cloud of object
    int *cloudPrj; //cloud projection point
} ImgInfo; // image information

typedef struct {
    int startIndex; // start index of working part
    int stepLength; // step length of working part
}PartInfo;

__device__ void rrefCudaKernal(float *matrix, MatrixCoor *matCoor, float zeroPar); //rref port function

__device__ void getElement(float *matrix, MatrixCoor *matCoor); //get element of matrix

__device__ void setElement(float *matrix, MatrixCoor *matCoor); //set element value of matrix

__device__ void STDWkEle(float *matrix, MatrixCoor *matCoor, int StdIndex, int rowIndex); //set every row's first element as 1

__device__ void STDRowSub(float *matrix, MatrixCoor *matCoor, int STDIndex, int SubIndex, float par); //subtract row(SubIndex) from row(STDIndex)

__device__ void ExchangeRow(float *matrix, MatrixCoor *matCoor, int row1, int row2); //exchange position of row1, row2

__device__ void TriangleTransDown(float *matrix, MatrixCoor *matCoor, int rowIndex, int set1Index); //matrix's triangle tranformation down

__device__ void TriangleTransUp(float *matrix, MatrixCoor *matCoor, int rowIndex, int set1Index); //matrix's triangle tranformation up

__device__ void LEqPEqPrjPointKernal(float *LEqArray, float *PEqArray, float *PrjArray);

__device__ void CoorCnvCudaKernal(float *coorXYZ, float *glCoor, float *resCoor);

__device__ void calWkCoorDirCudaKernal(float *coorRes, float *glRes); //calculate working coordinate's xyz dierction vector in global coordinate

__device__ void DirVector(float *startPoint, float *endPoint, float *res); //get direction vector from startPoint and endPoint, save result at res

__device__ float LenVector(float *vector); //get length of vector

__global__ void calWkCoorCloudPrj(int *prjCoor, float *cloud, float *coorRes, ImgInfo *imgInfo, int startIndex);

__global__ void calWkCoorImgCudaKernal(int *img, float *cloud, float *coorRes, ImgInfo *imgInfo);

__global__ void calWkCoorImgCudaKernalPart(int *img, float *cloud, float *coorRes, ImgInfo *imgInfo, PartInfo partInfo);

void calWkCoorImg(ImgInfo *imgInfo);

void calWkCoorImgPart(ImgInfo *imgInfo);
#endif // MATRIXBASE_H
