#include "matrixBase.h"

/*
 * cuda kernal of rref calculation
 * matrix: input matrix
 * matCoor: matrix information
 * zeroPar: zero parameter border
*/
__device__ void rrefCudaKernal(float *matrix, MatrixCoor *matCoor, float zeroPar) {
//    printf("kernal funcion is working\n");
//    STDWkEle(matrix, matCoor, 0, 0);
//    STDRowSub(matrix, matCoor, 0, 1, 1);
//    ExchangeRow(matrix, matCoor, 0, 2);
    int set1Index = 0; // set set1Index=0 as default
    for(int i = 0;i < matCoor->rows;i++) {
//        printf("row: %d\n", i);
        int exIncrease = 1; //set exchange increase=1 as default
        matCoor->col = set1Index;
//        printf("set1Index Init: %d\n", set1Index);
        matCoor->row = i;
        getElement(matrix, matCoor);
        if(matCoor->eleValue) {
//            printf("----------%d %d: %f\n", matCoor->row, matCoor->col, matCoor->eleValue);
            STDWkEle(matrix, matCoor, set1Index, i);
        } else {
//            printf("----error------%d %d: %f\n", matCoor->row, matCoor->col, matCoor->eleValue);
            /*row exchange calculation*/
            while ((abs(matCoor->eleValue) <= zeroPar)&&((i + exIncrease) < matCoor->rows)) {
                ExchangeRow(matrix, matCoor, i, i + exIncrease);
                getElement(matrix, matCoor);
                exIncrease++;
            }
            /*set1Index calculation*/
            matCoor->col = set1Index;
            matCoor->row = i;
            getElement(matrix, matCoor);
            while((abs(matCoor->eleValue) <= zeroPar)&&(set1Index < matCoor->cols)) {
//                printf("value: %d\n", matCoor->eleValue);
                set1Index++;
                matCoor->col = set1Index;
                matCoor->row = i;
                getElement(matrix, matCoor);
//                printf("----------%d %d: %f\n", matCoor->row, matCoor->col, matCoor->eleValue);
            }
//            printf("set1Index: %d\n", set1Index);
            if(set1Index < matCoor->cols - 1) {
                STDWkEle(matrix, matCoor, set1Index, i);
            }
        }
//        printf("----\n");
//        printf("set1Index: %d\n", set1Index);
//        printf("final set1Index: %d\n", set1Index);
        if(i < (matCoor->rows - 1)) {
            TriangleTransDown(matrix, matCoor, i, set1Index);
        }
        set1Index++;
    }
    /*clear nan error*/
    for(int i = 0;i < matCoor->rows;i++) {
        for(int j = 0;j < matCoor->cols;j++) {
            matCoor->row = i;
            matCoor->col = j;
            getElement(matrix, matCoor);
            if(isnan(matCoor->eleValue)) {
                matCoor->eleValue = 0;
                setElement(matrix, matCoor);
            }
        }
    }
    /*clear -0*/
    for(int i = 0;i < matCoor->rows;i++) {
        for(int j = 0;j < matCoor->cols;j++) {
            matCoor->row = i;
            matCoor->col = j;
            getElement(matrix, matCoor);
            if(matCoor->eleValue == -0) {
                matCoor->eleValue = 0;
                setElement(matrix, matCoor);
            }
        }
    }
    /*up triangle calculation*/
    for(int i = matCoor->rows - 1;i > 0;i--) {
        int wk1Index = 0; //set 1 element index = 0 as default
        matCoor->col = wk1Index;
        matCoor->row = i;
        getElement(matrix, matCoor);
        while((abs(matCoor->eleValue) <= zeroPar)&&(wk1Index < matCoor->cols)){
            wk1Index++;
            matCoor->col = wk1Index;
            matCoor->row = i;
            getElement(matrix, matCoor);
//            printf("i:%d index: %d value:%f\n", i, wk1Index, matCoor->eleValue);
            if (matCoor->eleValue) {
//                printf("not zero, %f", matCoor->eleValue);
            }
//            printf("%d\n", matCoor->cols);
        }
//        printf("----row:%d, wkIndex: %d\n", i, wk1Index);
        TriangleTransUp(matrix, matCoor, i, wk1Index);
    }

}

/*
 * get element from matrix, which coordinate is (row, col)
 * matrix: input matrix
 * matCoor: matrix coordinate information
*/
__device__ void getElement(float *matrix, MatrixCoor *matCoor) {
    unsigned int index = matCoor->row*matCoor->cols + matCoor->col;
    matCoor->eleValue = matrix[index];
}

/*
 * set element from matrix, which coordinate is (row, col)
 * matrix: input matrix
 * matCoor: matrix coordinate information
*/
__device__ void setElement(float *matrix, MatrixCoor *matCoor) {
    unsigned int index = matCoor->row*matCoor->cols + matCoor->col;
    matrix[index] = matCoor->eleValue;
}

/*
 * transform the first element as 1
 *
 * matrix: working matrix
 * matCoor: matrix information
 * StdIndex: index of set 1 element(0~rows)
 * rowIndex: set1Element index(0~cols)
*/
__device__ void STDWkEle(float *matrix, MatrixCoor *matCoor, int StdIndex, int rowIndex) {
    for(int i = rowIndex;i < matCoor->rows;i++) {
        matCoor->row = i;
        matCoor->col = StdIndex;
        getElement(matrix, matCoor);
        float divValue = matCoor->eleValue; //get divide value for standart element
//        printf("divValue: %f\n", divValue);
        if (!divValue) {
            continue;
        }
        for(int j = 0;j < matCoor->cols;j++) {
            matCoor->row = i;
            matCoor->col = j;
            getElement(matrix, matCoor);
            float wkValue = matCoor->eleValue; //get working element's value
            float setValue = wkValue/divValue; //get result of final working value
            matCoor->eleValue = setValue;
            setElement(matrix, matCoor);
        }
    }
}

/*
 * subtract row(SubIndex) from row(STDIndex)
 * matrix: working matrix
 * matCoor: matrix's information
 * STDIndex: standart row's index
 * SubIndex: row's index which will be subtracted
*/
__device__ void STDRowSub(float *matrix, MatrixCoor *matCoor, int STDIndex, int SubIndex, float par) {
    for(int i = 0;i < matCoor->cols;i++) {
        matCoor->col = i;
        matCoor->row = STDIndex;
        getElement(matrix, matCoor);
        float STDValue = matCoor->eleValue;
        matCoor->row = SubIndex;
        getElement(matrix, matCoor);
        float SubValue = matCoor->eleValue;
        float res = SubValue - STDValue*par;
//        printf("%f ", res);
        matCoor->eleValue = res;
        setElement(matrix, matCoor);
    }
}

/*
 * exchange row1 and row2
 * matrix: working matrix
 * matCoor: matrix information
 * row1: exchange row1 index
 * row2: exchange row2 index
*/
__device__ void ExchangeRow(float *matrix, MatrixCoor *matCoor, int row1, int row2) {
    for(int i = 0;i < matCoor->cols;i++) {
        matCoor->col = i;
        matCoor->row = row1;
        getElement(matrix, matCoor);
        float exchangeCache = matCoor->eleValue;
        matCoor->row = row2;
        getElement(matrix, matCoor);
        matCoor->row = row1;
        setElement(matrix, matCoor);
        matCoor->row = row2;
        matCoor->eleValue = exchangeCache;
        setElement(matrix, matCoor);
    }
}

/*
 * triangle transformation down
 * matrix: working matrix
 * matCoor: matrix information
 * rowIndex: working row's index
*/
__device__ void TriangleTransDown(float *matrix, MatrixCoor *matCoor, int rowIndex, int set1Index) {
    for(int i = rowIndex + 1;i < matCoor->rows;i++) {
        matCoor->row = i;
        matCoor->col = set1Index;
        getElement(matrix, matCoor);
        if(matCoor->eleValue) {
            STDRowSub(matrix, matCoor, rowIndex, i, 1);
        } else {
            continue;
        }
    }
}

/*
 * triangle transformation up
 * matrix: working matrix
 * matCoor: matrix information
 * rowIndex: working row's index
 * set1Index: set1 element's index
*/
__device__ void TriangleTransUp(
        float *matrix,
        MatrixCoor *matCoor,
        int rowIndex,
        int set1Index) {
    for(int i = rowIndex - 1;i >= 0;i--) {
        matCoor->col = set1Index;
        matCoor->row = i;
        getElement(matrix, matCoor);
        float par = matCoor->eleValue;
//        printf("par: %f ", par);
        STDRowSub(matrix, matCoor, rowIndex, i, par);
    }
}

__device__ void LEqPEqPrjPointKernal(float *LEqArray, float *PEqArray, float *PrjArray) {
//    printf("kernal funcion is working\n");
//    printf("%f\n", LEqArray[0]);
    float A0 = LEqArray[0];
    float B0 = LEqArray[1];
    float C0 = LEqArray[2];
    float x0 = LEqArray[3];
    float y0 = LEqArray[4];
    float z0 = LEqArray[5];

    float A1 = PEqArray[0];
    float B1 = PEqArray[1];
    float C1 = PEqArray[2];
    float x1 = PEqArray[3];
    float y1 = PEqArray[4];
    float z1 = PEqArray[5];

    float t = (A1*(x1 - x0) + B1*(y1 - y0) + C1*(z1 - z0))/(A0*A1 + B0*B1 + C0*C1);
    PrjArray[0] = A0*t + x0;
    PrjArray[1] = B0*t + y0;
    PrjArray[2] = C0*t + z0;

}

/*
 * convert glCoor to coorXYZ
 * coorXYZ: local axis direction vector
 * glCoor: global coordinate as input data
 * resCoor: result cache
*/
__device__ void CoorCnvCudaKernal(float *coorXYZ, float *glCoor, float *resCoor) {
//    printf("kernal is working\n");
    float *EquMatrix = (float*)malloc(sizeof(float)*12);

    /*set EquMatrix information*/
    MatrixCoor *matCoorEqu = (MatrixCoor*)malloc(sizeof(MatrixCoor));
    matCoorEqu->rows = 3;
    matCoorEqu->cols = 4;

    /*set coorXYZ director information*/
    MatrixCoor *matCoorXYZ = (MatrixCoor*)malloc(sizeof(MatrixCoor));
    matCoorXYZ->rows = 4;
    matCoorXYZ->cols = 3;
    /*standardize xyz axis*/
    for(int i = 0;i < 3;i++) {
        float axisLength = 0;
        for(int j =0;j < 3;j++) {
            matCoorXYZ->row = i;
            matCoorXYZ->col = j;
            getElement(coorXYZ, matCoorXYZ);
            axisLength += matCoorXYZ->eleValue*matCoorXYZ->eleValue;
        }
        for(int j =0;j < 3;j++) {
            matCoorXYZ->row = i;
            matCoorXYZ->col = j;
            getElement(coorXYZ, matCoorXYZ);
            matCoorXYZ->eleValue /= sqrt(axisLength);
            setElement(coorXYZ, matCoorXYZ);
        }
    }
    /*generate EquMatrix*/
    for(int i = 0;i < 3;i++) {
//        printf("%d\n", i);
        for(int j = 0;j < 3;j++) {
            matCoorXYZ->col = j;
            matCoorXYZ->row = i;
            getElement(coorXYZ, matCoorXYZ);
            matCoorEqu->eleValue = matCoorXYZ->eleValue;
            matCoorEqu->col = i;
            matCoorEqu->row = j;
            setElement(EquMatrix, matCoorEqu);
        }
        matCoorXYZ->col = i;
        matCoorXYZ->row = 3;
        getElement(coorXYZ, matCoorXYZ);
        matCoorEqu->eleValue = glCoor[i] - matCoorXYZ->eleValue;
        matCoorEqu->col = 3;
        matCoorEqu->row = i;
        setElement(EquMatrix, matCoorEqu);
    }
    for(int i = 0;i < 3;i++) {
        for(int j = 0;j < 4;j++) {
            matCoorEqu->row = i;
            matCoorEqu->col = j;
            getElement(EquMatrix, matCoorEqu);
//            printf("%f ", matCoorEqu->eleValue);
        }
//        printf("\n");
    }
    rrefCudaKernal(EquMatrix, matCoorEqu, 0.001);
//    printf("------\n");
    for(int i = 0;i < 3;i++) {
        for(int j = 0;j < 4;j++) {
            matCoorEqu->row = i;
            matCoorEqu->col = j;
            getElement(EquMatrix, matCoorEqu);
//            printf("%f ", matCoorEqu->eleValue);
        }
//        printf("\n");
    }
    for(int i = 0;i < 3;i++) {
        matCoorEqu->col = 3;
        matCoorEqu->row = i;
        getElement(EquMatrix, matCoorEqu);
        resCoor[i] = matCoorEqu->eleValue;
    }
}

/*
 * calculate wk coordinate vector in gl coor
 * coorRes: coordinate border information
 * glRes: calculation result
*/
__device__ void calWkCoorDirCudaKernal(float *coorRes, float *glRes) {
    /*
     * S3S2
     * X
     */
    int Cnt = 0;
    int borderStart = 2;
    int borderEnd = 1;
    for(int i = 0;i < 3;i++) {
        glRes[Cnt + i] = coorRes[borderEnd*3 + i + 1] - coorRes[borderStart*3 + i + 1];
    }
    /*
     * S3S4
     * Y
     */
    Cnt = 3;
    borderStart = 2;
    borderEnd = 3;
    for(int i = 0;i < 3;i++) {
        glRes[Cnt + i] = coorRes[borderEnd*3 + i + 1] - coorRes[borderStart*3 + i + 1];
    }
    /*
     * Z
     */
    glRes[6] = glRes[1]*glRes[5] - glRes[2]*glRes[4];
    glRes[7] = glRes[2]*glRes[3] - glRes[0]*glRes[5];
    glRes[8] = glRes[0]*glRes[4] - glRes[1]*glRes[3];
    /*
     * O coordinate in global coordinate
     */
    Cnt = 9;
    for(int i = 0;i < 3;i++) {
        glRes[Cnt + i] = coorRes[7 + i];
    }
}

/*
 * img: image float array converted from image matrix
 * cloud: working object's cloud
 * coorRes: camera's border coordinate result
 * imgInfo: image information
*/
__global__ void calWkCoorImgCudaKernal(int *img, float *cloud, float *coorRes, ImgInfo *imgInfo) {
//    printf("kernal starting %d\n", (int)cloud[0]);
    /*calculate xyz axis in global coordinate*/
    float *glRes = (float*)malloc(sizeof(float)*12);
    calWkCoorDirCudaKernal(coorRes, glRes);
//    for(int i = 0;i < 12;i++) {
//        if((i + 1) % 3) {
//            printf("%f ", glRes[i]);
//        } else {
//            printf("%f\n", glRes[i]);;
//        }
//    }
    /*get working cloud point*/
    int kernalIndex = (threadIdx.x + blockDim.x*blockIdx.x);
    int glWkIndex = kernalIndex * 3 + 1;
    if(kernalIndex >= ((int)(cloud[0]) - 1)) {
        printf("%f\n", kernalIndex);
        return;
    }
//    int glWkIndex = 1;
    float *wkCloudPoint = (float*)malloc(sizeof(float)*3);
    for(int i = 0;i < 3;i++) {
        wkCloudPoint[i] = cloud[glWkIndex + i];
    }

    /*calculate sensor size*/
    float senWidth = LenVector(glRes);
    float senHight = LenVector(glRes + 3);
//    printf("senWidth: %f senHeight: %f\n", senWidth, senHight);
//    printf("-----\n");
//    for(int i = 0;i < 12;i++) {
//        if((i + 1) % 3) {
//            printf("%f ", glRes[i]);
//        } else {
//            printf("%f\n", glRes[i]);;
//        }
//    }
    /*PEq standard equation*/
    float *PEq = (float*)malloc(sizeof(float)*6);
    for(int i = 0;i < 12;i++) {
        PEq[i] = glRes[6 + i];
    }

    /*LEq standard equation*/
    float *LEq = (float*)malloc(sizeof(float)*6);
    DirVector(coorRes + 25, wkCloudPoint, LEq);
    for(int i = 0;i < 3;i++) {
        LEq[i + 3] = wkCloudPoint[i];
    }

    /*LEq and PEq projection point, in out judge*/
    float *prjPoint = (float *)malloc(3*sizeof(float));
    float *prjPointLo = (float *)malloc(3*sizeof(float));
    LEqPEqPrjPointKernal(LEq, PEq, prjPoint);
    CoorCnvCudaKernal(glRes, prjPoint, prjPointLo);
    if((prjPointLo[0] < senWidth) && (prjPointLo[1] < senHight)) {
//        printf("pixel inside\n");
        int prjRow = imgInfo->rows - (int)(imgInfo->rows * (prjPointLo[1] / senHight));
        int prjCol = (int)(imgInfo->cols * (prjPointLo[0] / senWidth));
        for(int i = 0;i < 3;i++) {
            int imgWkIndex = i*imgInfo->cols*imgInfo->rows + prjRow*imgInfo->cols + prjCol;
            img[imgWkIndex] = 0;
        }
    }
//    int prjRow = imgInfo->rows - (int)(imgInfo->rows * (prjPointLo[1] / senHight));
//    int prjCol = (int)(imgInfo->cols * (prjPointLo[0] / senWidth));
//    for(int i = 0;i < 3;i++) {
//        int imgWkIndex = i*imgInfo->cols*imgInfo->rows + prjRow*imgInfo->cols + prjCol;
//        img[imgWkIndex] = 0;
//    }
//    printf("glIndex: %d\n", glWkIndex);
//    printf("gl: %f %f %f\n", prjPoint[0], prjPoint[1], prjPoint[2]);
//    printf("real: %f %f %f\n", prjPointLo[0], prjPointLo[1], prjPointLo[2]);
//    printf("pos: %d %d\n", imgInfo->rows, imgInfo->cols);
//    printf("pixle: %d %d\n", prjRow, prjCol);
}

/*
 * a
*/
__global__ void calWkCoorCloudPrj(int *cloudPrj, float *cloud, float *coorRes, ImgInfo *imgInfo, int startIndex) {
//    printf("kernal starting %d\n", startIndex);

//    __syncthreads();
    /*calculate xyz axis in global coordinate*/
    float *glRes = (float*)malloc(sizeof(float)*12);
    calWkCoorDirCudaKernal(coorRes, glRes);
//    for(int i = 0;i < 12;i++) {
//        if((i + 1) % 3) {
//            printf("%f ", glRes[i]);
//        } else {
//            printf("%f\n", glRes[i]);;
//        }
//    }

    /*get working cloud point*/
    int wkIndex = threadIdx.x + blockDim.x*blockIdx.x + startIndex;
    int glWkIndex = wkIndex * 3 + 1;
//    printf("partLen: %d startIndex: %d\n", (int)cloud[0], startIndex);
    if(wkIndex >= (int)cloud[0]) {
//        printf("--out--%d\n", glWkIndex);
        return;
    } else {
//        printf("%d\n", glWkIndex);
    }
    float *wkCloudPoint = (float*)malloc(sizeof(float)*3);
    for(int i = 0;i < 3;i++) {
        wkCloudPoint[i] = cloud[glWkIndex + i];
//        wkCloudPoint[i] = cloud[i];
//        printf("%f ", cloud[glWkIndex + i]);
    }
//    printf("\n");
    /*calculate sensor size*/
    float senWidth = LenVector(glRes);
    float senHight = LenVector(glRes + 3);
//    printf("senWidth: %f senHeight: %f\n", senWidth, senHight);
//    printf("-----\n");
//    for(int i = 0;i < 12;i++) {
//        if((i + 1) % 3) {
//            printf("%f ", glRes[i]);
//        } else {
//            printf("%f\n", glRes[i]);;
//        }
//    }
    /*PEq standard equation*/
    float *PEq = (float*)malloc(sizeof(float)*6);
    for(int i = 0;i < 12;i++) {
        PEq[i] = glRes[6 + i];
    }

    /*LEq standard equation*/
    float *LEq = (float*)malloc(sizeof(float)*6);
    DirVector(coorRes + 25, wkCloudPoint, LEq);
    for(int i = 0;i < 3;i++) {
        LEq[i + 3] = wkCloudPoint[i];
    }

    /*LEq and PEq projection point, in out judge*/
    float *prjPoint = (float *)malloc(3*sizeof(float));
    float *prjPointLo = (float *)malloc(3*sizeof(float));
    LEqPEqPrjPointKernal(LEq, PEq, prjPoint);
    CoorCnvCudaKernal(glRes, prjPoint, prjPointLo);
    if((prjPointLo[0] < senWidth) && (prjPointLo[1] < senHight)) {
//        printf("pixel inside\n");
        int prjRow = imgInfo->rows - (int)(imgInfo->rows * (prjPointLo[1] / senHight));
        int prjCol = (int)(imgInfo->cols * (prjPointLo[0] / senWidth));
        int cloudResWkIndex = wkIndex * 3;
        cloudPrj[cloudResWkIndex] = 1;
        cloudPrj[cloudResWkIndex + 1] = prjRow;
        cloudPrj[cloudResWkIndex + 2] = prjCol;
    } else {
        int cloudResWkIndex = wkIndex * 3;
        cloudPrj[cloudResWkIndex] = 0;
        cloudPrj[cloudResWkIndex + 1] = 0;
        cloudPrj[cloudResWkIndex + 2] = 0;
    }
//    __syncthreads();
//    int prjRow = imgInfo->rows - (int)(imgInfo->rows * (prjPointLo[1] / senHight));
//    int prjCol = (int)(imgInfo->cols * (prjPointLo[0] / senWidth));
//    for(int i = 0;i < 3;i++) {
//        int imgWkIndex = i*imgInfo->cols*imgInfo->rows + prjRow*imgInfo->cols + prjCol;
//        img[imgWkIndex] = 0;
//    }
//    printf("glIndex: %d\n", glWkIndex);
//    printf("gl: %f %f %f\n", prjPoint[0], prjPoint[1], prjPoint[2]);
//    printf("real: %f %f %f\n", prjPointLo[0], prjPointLo[1], prjPointLo[2]);
//    printf("pos: %d %d\n", imgInfo->rows, imgInfo->cols);
//    printf("pixle: %d %d\n", prjRow, prjCol);
}

/*
 * img: image float array converted from image matrix
 * cloud: working object's cloud
 * coorRes: camera's border coordinate result
 * imgInfo: image information
 * partInfo: cloud working part information
*/
__global__ void calWkCoorImgCudaKernalPart(int *img, float *cloud, float *coorRes, ImgInfo *imgInfo, PartInfo partInfo) {
//    printf("kernal starting\n");
    /*calculate xyz axis in global coordinate*/
    float *glRes = (float*)malloc(sizeof(float)*12);
    calWkCoorDirCudaKernal(coorRes, glRes);
//    for(int i = 0;i < 12;i++) {
//        if((i + 1) % 3) {
//            printf("%f ", glRes[i]);
//        } else {
//            printf("%f\n", glRes[i]);;
//        }
//    }

    /*get working cloud point*/
    int glWkIndex = (threadIdx.x + blockDim.x*blockIdx.x) * 3 + 1;
    float *wkCloudPoint = (float*)malloc(sizeof(float)*3);
    for(int i = 0;i < 3;i++) {
        wkCloudPoint[i] = cloud[glWkIndex + i];
    }

    /*calculate sensor size*/
    float senWidth = LenVector(glRes);
    float senHight = LenVector(glRes + 3);
//    printf("senWidth: %f senHeight: %f\n", senWidth, senHight);
//    printf("-----\n");
//    for(int i = 0;i < 12;i++) {
//        if((i + 1) % 3) {
//            printf("%f ", glRes[i]);
//        } else {
//            printf("%f\n", glRes[i]);;
//        }
//    }
    /*PEq standard equation*/
    float *PEq = (float*)malloc(sizeof(float)*6);
    for(int i = 0;i < 12;i++) {
        PEq[i] = glRes[6 + i];
    }

    /*LEq standard equation*/
    float *LEq = (float*)malloc(sizeof(float)*6);
    DirVector(coorRes + 25, wkCloudPoint, LEq);
    for(int i = 0;i < 3;i++) {
        LEq[i + 3] = wkCloudPoint[i];
    }

    /*LEq and PEq projection point, in out judge*/
    float *prjPoint = (float *)malloc(3*sizeof(float));
    float *prjPointLo = (float *)malloc(3*sizeof(float));
    LEqPEqPrjPointKernal(LEq, PEq, prjPoint);
    CoorCnvCudaKernal(glRes, prjPoint, prjPointLo);
    if((prjPointLo[0] < senWidth) && (prjPointLo[1] < senHight)) {
//        printf("pixel inside\n");
        int prjRow = imgInfo->rows - (int)(imgInfo->rows * (prjPointLo[1] / senHight));
        int prjCol = (int)(imgInfo->cols * (prjPointLo[0] / senWidth));
        for(int i = 0;i < 3;i++) {
            int imgWkIndex = i*imgInfo->cols*imgInfo->rows + prjRow*imgInfo->cols + prjCol;
            img[imgWkIndex] = 0;
        }
    }
//    int prjRow = imgInfo->rows - (int)(imgInfo->rows * (prjPointLo[1] / senHight));
//    int prjCol = (int)(imgInfo->cols * (prjPointLo[0] / senWidth));
//    for(int i = 0;i < 3;i++) {
//        int imgWkIndex = i*imgInfo->cols*imgInfo->rows + prjRow*imgInfo->cols + prjCol;
//        img[imgWkIndex] = 0;
//    }
//    printf("glIndex: %d\n", glWkIndex);
//    printf("gl: %f %f %f\n", prjPoint[0], prjPoint[1], prjPoint[2]);
//    printf("real: %f %f %f\n", prjPointLo[0], prjPointLo[1], prjPointLo[2]);
//    printf("pos: %d %d\n", imgInfo->rows, imgInfo->cols);
//    printf("pixle: %d %d\n", prjRow, prjCol);
}

/*
 * get direction vector from startPoint and endPoint, save result at res
 * startPoint: start point
 * endPoint: end point
 * res: result
*/
__device__ void DirVector(float *startPoint, float *endPoint, float *res) {
    for(int i = 0;i < 3;i++) {
        res[i] = endPoint[i] - startPoint[i];
    }
}

__device__ float LenVector(float *vector) {
    float res = 0;
//    for(int i = 0;i < 3;i++) {
//        printf("%f ", vector[i]);
//    }
//    printf("\n");
    for(int i = 0;i < 3;i++) {
        res += vector[i]*vector[i];
    }
    res = sqrt(res);
    return res;
}

void calWkCoorImg(ImgInfo *imgInfo) {
    /*camera border coordinate information*/
    float *coorResCuda; //coorRes to cuda
    cudaError_t erMark = cudaMalloc(&coorResCuda, sizeof(float)*28);
    printf("coorResInfo: %s\n", cudaGetErrorString(erMark));
    cudaMemcpy(coorResCuda, imgInfo->coorRes, sizeof(float)*28, cudaMemcpyHostToDevice);

    /*cloud projection point cache*/
    int *cloudPrjCuda;
    int cloudResCnt = (int)(imgInfo->cloud[0]);
    erMark = cudaMalloc(&cloudPrjCuda, sizeof(int)*cloudResCnt*3);
    printf("cloudPrjInfo: %s\n", cudaGetErrorString(erMark));

    /*image code data for cuda*/
//    int *imgCuda;
//    int pixelCnt = imgInfo->cols*imgInfo->rows*3;
//    cudaMalloc(&imgCuda, sizeof(int)*pixelCnt);
//    cudaMemcpy(imgCuda, imgInfo->img, sizeof(int)*pixelCnt, cudaMemcpyHostToDevice);

    /*cloud point for cuda*/
    float *cloudCuda;
    int cloudCnt = (3*(int)(imgInfo->cloud[0]) + 1);
//    int cloudCnt = 301;
    erMark = cudaMalloc(&cloudCuda, sizeof(float)*cloudCnt);
    printf("cloudInfo: %s\n", cudaGetErrorString(erMark));
    cudaMemcpy(cloudCuda, imgInfo->cloud, sizeof(float)*cloudCnt, cudaMemcpyHostToDevice);

    /*image information for cuda*/
    ImgInfo *imgInfoCuda;
    cudaMalloc(&imgInfoCuda, sizeof(ImgInfo));
    cudaMemcpy(imgInfoCuda, imgInfo, sizeof(imgInfo), cudaMemcpyHostToDevice);

    /*cloud projection calculation*/
    int perThread = 128;
    int threadCnt = ((int)(imgInfo->cloud[0]))/(perThread) + 1;
    int blockLen = 30; // set block div length
//    int threadCnt = ((int)(imgInfo->cloud[0]))/(perThread*blockLen) + 1;
    printf("all:%d threadCnt: %d\n", (int)(imgInfo->cloud[0]), threadCnt);
//    printf("all:%f threadCnt: %d\n", imgInfo->cloud[0], threadCnt);
//    calWkCoorImgCudaKernal<<<threadCnt, perThread>>>(imgCuda, cloudCuda, coorResCuda, imgInfoCuda);
//    calWkCoorImgCudaKernal<<<1, perThread>>>(imgCuda, cloudCuda, coorResCuda, imgInfoCuda);
//    calWkCoorCloudPrj<<<threadCnt, perThread>>>(cloudPrjCuda, cloudCuda, coorResCuda, imgInfoCuda, 0);
    calWkCoorCloudPrj<<<blockLen, perThread>>>(cloudPrjCuda, cloudCuda, coorResCuda, imgInfoCuda, 0);
//    for(int i = 0;i < threadCnt;i++) {
////        printf("partIndex: %d\n", i);

//        int startIndex = i * blockLen * perThread;
//        calWkCoorCloudPrj<<<blockLen, perThread>>>(cloudPrjCuda, cloudCuda, coorResCuda, imgInfoCuda, startIndex);
//        cudaError_t cudaEPr = cudaGetLastError();
//        printf("ker: %s\n", cudaGetErrorString(cudaEPr));
////        cudaMemcpy(imgInfo->cloudPrj + startIndex, cloudPrjCuda + startIndex, sizeof(int)*blockLen*perThread, cudaMemcpyDeviceToHost);
////        cudaMemcpy(imgInfo->cloudPrj, cloudPrjCuda, sizeof(int)*cloudResCnt, cudaMemcpyDeviceToHost);
//        cudaDeviceSynchronize();
//    }

//    cudaError_t cudaEPr = cudaGetLastError();
//    printf("%s\n", cudaGetErrorName(cudaEPr));
//    calWkCoorImgCudaKernal<<<1, 1>>>(imgCuda, cloudCuda, coorResCuda, imgInfoCuda);

    cudaMemcpy(imgInfo->cloudPrj, cloudPrjCuda, sizeof(int)*cloudResCnt, cudaMemcpyDeviceToHost);
//    cudaMemcpy(imgInfo->img, imgCuda, sizeof(int)*pixelCnt, cudaMemcpyDeviceToHost);

    /*release GPU mem*/
//    cudaFree(coorResCuda);
////    cudaFree(imgCuda);
//    cudaFree(cloudPrjCuda);
//    cudaFree(cloudCuda);
//    cudaFree(imgInfoCuda);

    erMark = cudaDeviceReset();
    printf("resetInfo: %s\n", cudaGetErrorString(erMark));
//    erMark = cudaDeviceSynchronize();
//    printf("synchroInfo: %s\n", cudaGetErrorString(erMark));
}

void calWkCoorImgPart(ImgInfo *imgInfo) {

    int partPointCnt = imgInfo->blockCnt * imgInfo->perThread;
    int glStartIndex = partPointCnt * imgInfo->startIndex;
    /*camera border coordinate information*/
    float *coorResCuda; //coorRes to cuda
    cudaError_t erMark = cudaMalloc(&coorResCuda, sizeof(float)*28);
//    printf("coorResInfo: %s\n", cudaGetErrorString(erMark));
    cudaMemcpy(coorResCuda, imgInfo->coorRes, sizeof(float)*28, cudaMemcpyHostToDevice);

    /*cloud projection point cache*/
    int *cloudPrjCuda;
    int cloudResCnt = partPointCnt;
    erMark = cudaMalloc(&cloudPrjCuda, sizeof(int)*cloudResCnt*3);
//    printf("cloudPrjInfo: %s\n", cudaGetErrorString(erMark));

    /*image code data for cuda*/
//    int *imgCuda;
//    int pixelCnt = imgInfo->cols*imgInfo->rows*3;
//    cudaMalloc(&imgCuda, sizeof(int)*pixelCnt);
//    cudaMemcpy(imgCuda, imgInfo->img, sizeof(int)*pixelCnt, cudaMemcpyHostToDevice);

    /*cloud point for cuda*/
    float *cloudCuda;
    int cloudCnt = partPointCnt*3 + 1;
//    int cloudCnt = 301;
    erMark = cudaMalloc(&cloudCuda, sizeof(float)*cloudCnt);
//    printf("cloudInfo: %s\n", cudaGetErrorString(erMark));
    cudaMemcpy(cloudCuda + 1, imgInfo->cloud + glStartIndex*3 + 1, sizeof(float)*partPointCnt*3, cudaMemcpyHostToDevice);
    cudaMemcpy(cloudCuda, imgInfo->cloud, sizeof(float), cudaMemcpyHostToDevice);
    /*image information for cuda*/
    ImgInfo *imgInfoCuda;
    cudaMalloc(&imgInfoCuda, sizeof(ImgInfo));
    cudaMemcpy(imgInfoCuda, imgInfo, sizeof(imgInfo), cudaMemcpyHostToDevice);

    /*cloud projection calculation*/
    int perThread = imgInfo->perThread;
//    int threadCnt = ((int)(imgInfo->cloud[0]))/(perThread) + 1;
    int blockLen = imgInfo->blockCnt; // set block div length
//    int threadCnt = ((int)(imgInfo->cloud[0]))/(perThread*blockLen) + 1;
//    printf("all:%d threadCnt: %d\n", (int)(imgInfo->cloud[0]), threadCnt);
//    printf("all:%f threadCnt: %d\n", imgInfo->cloud[0], threadCnt);
//    calWkCoorImgCudaKernal<<<threadCnt, perThread>>>(imgCuda, cloudCuda, coorResCuda, imgInfoCuda);
//    calWkCoorImgCudaKernal<<<1, perThread>>>(imgCuda, cloudCuda, coorResCuda, imgInfoCuda);
//    calWkCoorCloudPrj<<<threadCnt, perThread>>>(cloudPrjCuda, cloudCuda, coorResCuda, imgInfoCuda, 0);
//    int partPointCnt = imgInfo->blockCnt * imgInfo->perThread;
//    int glStartIndex = partPointCnt * imgInfo->startIndex;
    calWkCoorCloudPrj<<<blockLen, perThread>>>(cloudPrjCuda, cloudCuda, coorResCuda, imgInfoCuda, 0);
//    for(int i = 0;i < threadCnt;i++) {
////        printf("partIndex: %d\n", i);

//        int startIndex = i * blockLen * perThread;
//        calWkCoorCloudPrj<<<blockLen, perThread>>>(cloudPrjCuda, cloudCuda, coorResCuda, imgInfoCuda, startIndex);
//        cudaError_t cudaEPr = cudaGetLastError();
//        printf("ker: %s\n", cudaGetErrorString(cudaEPr));
////        cudaMemcpy(imgInfo->cloudPrj + startIndex, cloudPrjCuda + startIndex, sizeof(int)*blockLen*perThread, cudaMemcpyDeviceToHost);
////        cudaMemcpy(imgInfo->cloudPrj, cloudPrjCuda, sizeof(int)*cloudResCnt, cudaMemcpyDeviceToHost);
//        cudaDeviceSynchronize();
//    }

//    cudaError_t cudaEPr = cudaGetLastError();
//    printf("%s\n", cudaGetErrorName(cudaEPr));
//    calWkCoorImgCudaKernal<<<1, 1>>>(imgCuda, cloudCuda, coorResCuda, imgInfoCuda);

    cudaMemcpy(imgInfo->cloudPrj, cloudPrjCuda, sizeof(int)*partPointCnt*3, cudaMemcpyDeviceToHost);
//    cudaMemcpy(imgInfo->img, imgCuda, sizeof(int)*pixelCnt, cudaMemcpyDeviceToHost);

    /*release GPU mem*/
//    cudaFree(coorResCuda);
////    cudaFree(imgCuda);
//    cudaFree(cloudPrjCuda);
//    cudaFree(cloudCuda);
//    cudaFree(imgInfoCuda);

    erMark = cudaDeviceReset();
//    printf("resetInfo: %s\n", cudaGetErrorString(erMark));
//    erMark = cudaDeviceSynchronize();
//    printf("synchroInfo: %s\n", cudaGetErrorString(erMark));
}




