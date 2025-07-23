#include "imagesimctrl.h"
#include "ui_imagesimctrl.h"

ImageSimCtrl::ImageSimCtrl(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ImageSimCtrl)
{
    ui->setupUi(this);
    connect(ui->pushButton_startSim,
            &QPushButton::clicked,
            this,
            &ImageSimCtrl::startSim);
    connect(ui->pushButton_Differ,
            &QPushButton::clicked,
            this,
            &ImageSimCtrl::BackGrdDiffer);
    connect(ui->pushButton_DeteSim,
            &QPushButton::clicked,
            this,
            &ImageSimCtrl::startImageDetection);
//    qDebug() << QDir::currentPath();
    QFile *file = new QFile(QDir::currentPath() + "/dataInfo/dataInfo.txt");
//    qDebug() << file->exists();
    file->open(QIODevice::ReadOnly);
    QByteArray dataPathByte = file->readAll();
    file->close();
    delete file;
    this->dataPath = QString(dataPathByte);
//    qDebug() << QString(dataPathByte);
    ui->label_DataPath->setText(this->dataPath);
    ui->lineEdit_blockCnt->setText("65");
    ui->lineEdit_perThread->setText("128");
    ui->lineEdit_pixelDiff->setText("30");
    ui->lineEdit_trSize->setText("2");
    ui->lineEdit_threshold->setText("0.98");
}

void ImageSimCtrl::setDataPath(QString dataPath) {
    this->dataPath = dataPath;
    ui->label_DataPath->setText(this->dataPath);
}

ImageSimCtrl::~ImageSimCtrl()
{
    delete ui;
}

void ImageSimCtrl::startSim() {
    QDir *dir = new QDir();
    dir->setPath(this->dataPath + "/CamPar");
    QStringList CamNames = dir->entryList(QDir::Files);
//    qDebug() << CamNames;
    ImgInfo *imgInfo = (ImgInfo*)malloc(sizeof(ImgInfo));
    imgInfo->perThread = ui->lineEdit_perThread->text().toInt();
    imgInfo->blockCnt = ui->lineEdit_blockCnt->text().toInt();
    int pointStepCnt = imgInfo->perThread * imgInfo->blockCnt * 1;
    imgInfo->coorRes = (float*)malloc(sizeof(float)*28);
    imgInfo->cloud = (float*)malloc(sizeof(float)*(pointStepCnt*3 + 1));
//    imgInfo->img = (int*)malloc(sizeof(int)*INT_MAX);
    imgInfo->cloudPrj = (int *)malloc(sizeof(int)*(pointStepCnt*3));
    int *borderInfo = (int *)malloc(sizeof(int)*4);
    for(QString CamInfo : CamNames) {
//        qDebug() << CamInfo;
//        qDebug() << this->dataPath;
        QString fileName = this->dataPath + "/CamPar/" + CamInfo;
//        qDebug() << fileName;
//        ImgInfo *imgInfo = (ImgInfo*)malloc(sizeof(ImgInfo));
//        imgInfo->coorRes = (float*)malloc(sizeof(float)*28);
//        imgInfo->cloud = (float*)malloc(sizeof(float)*INT_MAX);
        this->getLineInfo(fileName, imgInfo->coorRes);
//        imgInfo->img = (int*)malloc(sizeof(int)*INT_MAX);
        dir->setPath(this->dataPath);
        QStringList filter;
        filter << "frame*";
        QStringList frameArray = dir->entryList(filter);
//        qDebug() << frameArray;
        filter.clear();
        for(QString frame:frameArray) {
            QString imgPath = this->dataPath + "/" + frame + "/" + CamInfo + ".png";
//            qDebug() << imgPath;
            cv::Mat frameCamImg = cv::imread(imgPath.toStdString()); // get working frame image's information
            cv::Mat resMat(frameCamImg.rows, frameCamImg.cols, CV_8UC3); // cloud image result cache
            imgInfo->rows = frameCamImg.rows;
            imgInfo->cols = frameCamImg.cols;
            for(int row = 0;row < resMat.rows;row++) {
                for(int col = 0;col < resMat.cols;col++) {
                    for(int i = 0;i < 3;i++) {
                        resMat.ptr<cv::Vec3b>(row)[col][i] = 255;
                    }
                }
            }
//            this->codeImg(resMat, imgInfo);
            QString cloudPath = this->dataPath + "/" + frame + "/StdObj";
            dir->setPath(cloudPath);
            filter << "HG_*";
            QStringList objList = dir->entryList(filter);
//            qDebug() << objList;

            for(QString objName : objList) {
                dir->setPath(this->dataPath + "/objTemImg/" + objName);
                if(!dir->exists()) {
                    dir->mkpath(this->dataPath + "/objTemImg/" + objName);
                }
                bool isInitObj = true;
                QString objPath = cloudPath + "/" + objName;
                qDebug() << objPath;
                dir->setPath(objPath);
                QStringList partList = dir->entryList(filter);
                for(QString partName : partList) {
                    QString partPath = objPath + "/" + partName;
//                    qDebug() << partPath;
                    int allPointCnt = this->getFileInfo(partPath);
                    int filePartCnt = allPointCnt/pointStepCnt + 1;
                    for (int i = 0;i < filePartCnt;i++) {
                        this->getLineInfo(partPath, imgInfo->cloud, i * imgInfo->blockCnt*imgInfo->perThread, imgInfo->blockCnt*imgInfo->perThread);
//                        qDebug() << imgInfo->cloud[0];
//                        calWkCoorImg(imgInfo);
                        int divParts = (pointStepCnt)/(imgInfo->blockCnt * imgInfo->perThread) + 1;
//                        qDebug() << divParts;
                        for(int i = 0;i < divParts;i++) {
                            imgInfo->startIndex = i;
                            calWkCoorImgPart(imgInfo);
                            int writeLen;
                            if (i < (divParts - 1)) {
                                writeLen = imgInfo->blockCnt*imgInfo->perThread;
                            } else {
                                writeLen = pointStepCnt % (imgInfo->blockCnt*imgInfo->perThread);
//                                writeLen = imgInfo->blockCnt*imgInfo->perThread;
                            }
                            for(int i = 0;i < writeLen;i += 3) {
                                if(imgInfo->cloudPrj[i]) {
//                                    qDebug() << imgInfo->cloudPrj[i + 1] << imgInfo->cloudPrj[i + 2];
                                    for(int j = 0;j < 3;j++) {
                                        resMat.ptr<cv::Vec3b>(imgInfo->cloudPrj[i + 1])[imgInfo->cloudPrj[i + 2]][j] = 0;
                                        if(isInitObj) {
                                            borderInfo[0] = imgInfo->cloudPrj[i + 1];
                                            borderInfo[1] = imgInfo->cloudPrj[i + 1];
                                            borderInfo[2] = imgInfo->cloudPrj[i + 2];
                                            borderInfo[3] = imgInfo->cloudPrj[i + 2];
                                            isInitObj = false;
                                        } else {
                                            this->getBorder(imgInfo->cloudPrj + i + 1, borderInfo);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                int imgRows = borderInfo[1] - borderInfo[0];
                int imgCols = borderInfo[3] - borderInfo[2];
                cv::Mat objTemImg(imgRows, imgCols, CV_8UC3);
                this->getObjTemImg(&resMat, &objTemImg, borderInfo);
                QString temImgPath = this->dataPath + "/objTemImg/" + objName + "/" + CamInfo + "_" + frame.mid(5) + ".png";
                cv::imwrite(temImgPath.toStdString(), objTemImg);
            }
//            this->decodeImg(&resMat, imgInfo);
            QString resWrPath = this->dataPath + "/cloudSim/" + CamInfo + "_" + frame.mid(5) + ".png";
            cv::imwrite(resWrPath.toStdString(), resMat);
        }
    }
    free(borderInfo);
    free(imgInfo);
}

void ImageSimCtrl::getLineInfo(QString fileName, float *coorRes) {
    QFile *file = new QFile(fileName);
//    qDebug() << fileName;
    file->open(QIODevice::ReadOnly);
    QString infoStr = QString(file->readAll());
    file->close();
    delete file;
//    qDebug() << infoStr;
    QStringList infoList;
    int startIndex = 0; // start index of single information
    int infoLen = 0; // information length from single line
    for(int i = 0;i < infoStr.length();i++) {
        if(infoStr.at(i) == "\n") {
            infoList << infoStr.mid(startIndex, infoLen);
            startIndex += infoLen + 1;
            infoLen = 0;
        } else {
            infoLen++;
        }
    }
//    qDebug() << infoList;
//    coorRes = (float*)realloc(coorRes, sizeof(float)*(infoList.length()*4 + 1)); // coordinate result's cache
    coorRes[0] = infoList.length(); // record length of result
    int glIndex = 1;
    int *spaceIndices = (int*)malloc(sizeof(int)*3); // index array
    for(QString info : infoList) {
        int markIndex = 0; // space mark index
        for(int i = 0;i < info.length();i++) {
            if(info.at(i) == " ") {
                if(markIndex < 3) {
                    spaceIndices[markIndex++] = i;
                } else {
                    break;
                }
            }
        }
//        for(int i = 0;i < 3;i++) {
//            qDebug() << spaceIndices[i];
//        }
        for(int i = 0;i < 3;i++) {
            if (i < 2) {
//                qDebug() << info.mid(spaceIndices[i] + 1, spaceIndices[i + 1] - spaceIndices[i] - 1);
                coorRes[glIndex++] = info.mid(spaceIndices[i] + 1, spaceIndices[i + 1] - spaceIndices[i] - 1).toFloat();
            } else {
//                qDebug() << info.mid(spaceIndices[i] + 1);
                coorRes[glIndex++] = info.mid(spaceIndices[i] + 1).toFloat();
            }
        }
    }
    free(spaceIndices);
}

void ImageSimCtrl::getLineInfo(QString fileName, float *coorRes, int startCnt, int infoCnt) {
    QFile *file = new QFile(fileName);
//    qDebug() << fileName;
    file->open(QIODevice::ReadOnly);
    QString infoStr = QString(file->readAll());
    file->close();
    delete file;
//    qDebug() << infoStr;
    QStringList infoList;
    int startIndex = 0; // start index of single information
    int infoLen = 0; // information length from single line
    for(int i = 0;i < infoStr.length();i++) {
        if(infoStr.at(i) == "\n") {
            infoList << infoStr.mid(startIndex, infoLen);
            startIndex += infoLen + 1;
            infoLen = 0;
        } else {
            infoLen++;
        }
    }
//    qDebug() << infoList;
//    coorRes = (float*)realloc(coorRes, sizeof(float)*(infoList.length()*4 + 1)); // coordinate result's cache
    coorRes[0] = infoList.length(); // record length of result
    int glIndex = 1;
    int *spaceIndices = (int*)malloc(sizeof(int)*3); // index array
    for(int i = 0;(i < infoCnt) && ((startCnt + i) < infoList.length());i++) {
        QString info = infoList[startCnt + i];
        int markIndex = 0; // space mark index
        for(int i = 0;i < info.length();i++) {
            if(info.at(i) == " ") {
                if(markIndex < 3) {
                    spaceIndices[markIndex++] = i;
                } else {
                    break;
                }
            }
        }
//        for(int i = 0;i < 3;i++) {
//            qDebug() << spaceIndices[i];
//        }
        for(int i = 0;i < 3;i++) {
            if (i < 2) {
//                qDebug() << info.mid(spaceIndices[i] + 1, spaceIndices[i + 1] - spaceIndices[i] - 1);
                coorRes[glIndex++] = info.mid(spaceIndices[i] + 1, spaceIndices[i + 1] - spaceIndices[i] - 1).toFloat();
            } else {
//                qDebug() << info.mid(spaceIndices[i] + 1);
                coorRes[glIndex++] = info.mid(spaceIndices[i] + 1).toFloat();
            }
        }
    }
    free(spaceIndices);
}

int ImageSimCtrl::getFileInfo(QString fileName) {
    QFile *file = new QFile(fileName);
    file->open(QIODevice::ReadOnly);
    QString infoStr = QString(file->readAll());
    file->close();
//    qDebug() << infoStr;
    QStringList infoList;
    int startIndex = 0; // start index of single information
    int infoLen = 0; // information length from single line
    for(int i = 0;i < infoStr.length();i++) {
        if(infoStr.at(i) == "\n") {
            infoList << infoStr.mid(startIndex, infoLen);
            startIndex += infoLen + 1;
            infoLen = 0;
        } else {
            infoLen++;
        }
    }
//    qDebug() << infoList;
    delete file;
    return infoList.length();
}

/*
 * get working coordinate's directory vector from coorRes
 * coorRes: coordinate result about camera information
 * dirRes: algorithm result from coorRes [x, y, z, o]
*/
void ImageSimCtrl::getWkImg(ImgInfo *imgInfo) {
    calWkCoorImg(imgInfo);
}

void ImageSimCtrl::codeImg(QString imgPath, ImgInfo *imgInfo) {
    cv::Mat wkImg = cv::imread(imgPath.toStdString());
    imgInfo->rows = wkImg.rows;
    imgInfo->cols = wkImg.cols;
    int imgCodeIndex = 0; // image code index, start at 0
    for(int i = 0;i < 3;i++) {
        for(int row = 0;row < imgInfo->rows;row++) {
            for(int col = 0;col < imgInfo->cols;col++) {
                imgInfo->img[imgCodeIndex++] = wkImg.ptr<cv::Vec3b>(row)[col][i];
//                qDebug() << imgCodeIndex;
            }
        }
    }
//    qDebug() << imgCodeIndex;
}

void ImageSimCtrl::codeImg(cv::Mat wkImg, ImgInfo *imgInfo) {
    imgInfo->rows = wkImg.rows;
    imgInfo->cols = wkImg.cols;
    int imgCodeIndex = 0; // image code index, start at 0
    for(int i = 0;i < 3;i++) {
        for(int row = 0;row < imgInfo->rows;row++) {
            for(int col = 0;col < imgInfo->cols;col++) {
                imgInfo->img[imgCodeIndex++] = wkImg.ptr<cv::Vec3b>(row)[col][i];
//                qDebug() << imgCodeIndex;
            }
        }
    }
//    qDebug() << imgCodeIndex;
}

void ImageSimCtrl::decodeImg(cv::Mat *imgRes, ImgInfo *imgInfo) {
    for(int i = 0;i < 3;i++) {
        for(int row = 0;row < imgInfo->rows;row++) {
            for(int col = 0;col < imgInfo->cols;col++) {
                int imgWkIndex = i*imgInfo->cols*imgInfo->rows + row*imgInfo->cols + col;
//                qDebug() << imgInfo->img[imgWkIndex];
                imgRes->ptr<cv::Vec3b>(row)[col][i] = (uchar)imgInfo->img[imgWkIndex];
            }
        }
    }
//    qDebug() << "f";
}

void ImageSimCtrl::decodeImg(cv::Mat *imgRes, float *deteRes, ImgInfo *imgInfo, float *decodeRes) {
    decodeRes[0] = 0;
    for(int row = 0;row < imgInfo->rows;row++) {
        for(int col = 0;col < imgInfo->cols;col++) {
            int wkIndex = row*imgInfo->cols + col;
//            qDebug() << deteRes[wkIndex];
            if(deteRes[wkIndex] >= ui->lineEdit_threshold->text().toFloat()) {
                if(decodeRes[0] < deteRes[wkIndex]) {
                    decodeRes[0] = deteRes[wkIndex];
                    decodeRes[1] = row;
                    decodeRes[2] = col;
                }
                for(int i = 0;i < 3;i++) {
                    imgRes->ptr<cv::Vec3b>(row)[col][i] = 0;
                }
            } else {
                for(int i = 0;i < 3;i++) {
                    imgRes->ptr<cv::Vec3b>(row)[col][i] = 255;
                }
            }
        }
    }
}

/*
 * get borderInfo and write it in borderInfo
 * wkPos: row col
 * borderInfo: rowMin rowMax colMin colMax
*/
void ImageSimCtrl::getBorder(int *wkPos, int *borderInfo) {
    if(borderInfo[0] > wkPos[0]) {
        borderInfo[0] = wkPos[0];
    }
    if(borderInfo[1] < wkPos[0]) {
        borderInfo[1] = wkPos[0];
    }
    if(borderInfo[2] > wkPos[1]) {
        borderInfo[2] = wkPos[1];
    }
    if(borderInfo[3] < wkPos[1]) {
        borderInfo[3] = wkPos[1];
    }
}

void ImageSimCtrl::getObjTemImg(cv::Mat *img, cv::Mat *temObj, int *borderInfo) {
    for(int i = 0;(i + borderInfo[0]) < borderInfo[1];i++) {
        for(int j = 0;(j + borderInfo[2]) < borderInfo[3];j++) {
            for(int k = 0;k < 3;k++) {
                temObj->ptr<cv::Vec3b>(i)[j][k] = img->ptr<cv::Vec3b>(i + borderInfo[0])[j + borderInfo[2]][k];
            }
        }
    }
}

/*
 * background differ algorithm
*/
void ImageSimCtrl::BackGrdDiffer() {
    QDir *dir = new QDir();
    dir->setPath(this->dataPath + "/CamPar");
    QStringList CamNames = dir->entryList(QDir::Files);
//    qDebug() << CamNames;
    for(QString CamInfo : CamNames) {
//        qDebug() << CamInfo;
        dir->setPath(this->dataPath);
        QStringList filter;
        filter << "frame*";
        QStringList frameArray = dir->entryList(filter);
//        qDebug() << frameArray;
        filter.clear();
        for(QString frame:frameArray) {
            QString imgPath = this->dataPath + "/" + frame + "/" + CamInfo + ".png";
            qDebug() << imgPath;
            QString StdImgPath = this->dataPath + "/StdImage/" + CamInfo + ".png";
            cv::Mat ObjImg = cv::imread(imgPath.toStdString());
            cv::Mat StdImg = cv::imread(StdImgPath.toStdString());
            cv::Mat diffRes(ObjImg.rows, ObjImg.cols, CV_8UC3); // differ image's result
            for(int row = 0;row < ObjImg.rows;row++) {
                for(int col = 0;col < ObjImg.cols;col++) {
                    int judge = 0;
                    for(int i = 0;i < 3;i++) {
                        judge += abs(ObjImg.ptr<cv::Vec3b>(row)[col][i] - StdImg.ptr<cv::Vec3b>(row)[col][i]);
                    }
                    if(judge >= ui->lineEdit_pixelDiff->text().toInt()) {
                        for(int i = 0;i < 3;i++) {
                            diffRes.ptr<cv::Vec3b>(row)[col][i] = 0;
                        }
                    } else {
                        for(int i = 0;i < 3;i++) {
                            diffRes.ptr<cv::Vec3b>(row)[col][i] = 255;
                        }
                    }
                }
            }
            QString diffResPath = this->dataPath + "/DifferFrame/" + CamInfo + "_" + frame.mid(5) + ".png";
            cv::imwrite(diffResPath.toStdString(), diffRes);
        }
    }
}

void ImageSimCtrl::startImageDetection() {
    QString diffFramePath = this->dataPath + "/CamPar";
    QDir *dir = new QDir();
    dir->setPath(diffFramePath);
    QStringList filter;
    QStringList CamNames = dir->entryList(QDir::Files);
//    qDebug() << CamNames;
    QString objModPath = this->dataPath + "/objTemImg";
//    qDebug() << objModPath;
    dir->setPath(objModPath);
    filter << "HG_*";
    QStringList objNames = dir->entryList(filter);
//    qDebug() << objNames;
    filter.clear();
    filter << "frame*";
    dir->setPath(this->dataPath);
    QStringList frameArray = dir->entryList(filter);
    filter.clear();
    int frameCnt = frameArray.length();
//    qDebug() << frameCnt;
    for(QString objName : objNames) {
        QString objModImgPath = objModPath + "/" + objName;
        dir->setPath(objModImgPath);
        for(QString CamName : CamNames) {
            filter << CamName + "*";
            QStringList modImgs = dir->entryList(filter);
            filter.clear();
//            qDebug() << modImgs;
            for(int i = 0;i < frameCnt;i++) {
                for(QString modImg : modImgs) {
                    QString objImgPath = objModImgPath + "/" + modImg;
                    qDebug() << objImgPath;
                    int undlineMark = 0;
                    for(int j = 0;j < modImg.length();j++) {
                        if(modImg.at(j) == "_") {
                            undlineMark = j;
                            break;
                        }
                    }
                    QString wkFramePath = this->dataPath + "/DifferFrame/" + CamName + "_" + QString::number(i + 1) + ".png";
                    qDebug() << wkFramePath;
//                    qDebug() << "--------";
                    cv::Mat objImg = cv::imread(objImgPath.toStdString());
                    ImgInfo objImgInfo;
                    objImgInfo.img = (int*)malloc(sizeof(int)*objImg.cols*objImg.rows*3);
                    this->codeImg(objImg, &objImgInfo);

                    cv::Mat wkFrameImg = cv::imread(wkFramePath.toStdString());
                    ImgInfo wkFrameInfo;
                    float *deteRes = (float*)malloc(sizeof(float)*wkFrameImg.cols*wkFrameImg.rows);
                    wkFrameInfo.img = (int*)malloc(sizeof(int)*wkFrameImg.cols*wkFrameImg.rows*3);
                    this->codeImg(wkFrameImg, &wkFrameInfo);
                    DeteInfo deteInfo;
                    deteInfo.trSize = ui->lineEdit_trSize->text().toInt();
                    deteInfo.threshold = ui->lineEdit_threshold->text().toFloat();
                    deteInfo.pixelDiff = ui->lineEdit_pixelDiff->text().toInt();
                    this->ImageDetectionCuda(&objImgInfo, &wkFrameInfo, deteRes, deteInfo);

                    float *decodeRes = (float*)malloc(sizeof(float)*3);
                    cv::Mat resImg(wkFrameImg.rows, wkFrameImg.cols, CV_8UC3);
                    this->decodeImg(&resImg, deteRes, &wkFrameInfo, decodeRes);
                    QString judgePath = this->dataPath + "/judgeRes/" + objName;
                    qDebug() << judgePath;
//                    qDebug() << "result: " << decodeRes[0] << decodeRes[1] << decodeRes[2];
                    qDebug() << "-------";
                    dir->setPath(judgePath);
                    if (!dir->exists()) {
                        dir->mkpath(judgePath);
                    }
                    if(decodeRes[0]) {
                        QString judgePathRes = judgePath + "/" + CamName + "_" + QString::number(i + 1) + "_" + modImg.mid(undlineMark + 1, 1) + ".png";
                        QString deteImgResPath = judgePath + "/" + CamName + "_" + QString::number(i + 1) + "_" + modImg.mid(undlineMark + 1, 1) + "_res" + ".png";
                        int *posSize = (int*)malloc(sizeof(int)*4);
                        posSize[0] = decodeRes[1];
                        posSize[1] = decodeRes[2];
                        posSize[2] = objImgInfo.rows;
                        posSize[3] = objImgInfo.cols;
                        this->setRecToImg(&wkFrameImg, posSize);
                        cv::imwrite(deteImgResPath.toStdString(), wkFrameImg);
                        cv::imwrite(judgePathRes.toStdString(), resImg);
                        break;
                    }
                }
            }

        }
    }
}

/*
 * set rectangle to img
 * img: working image
 * posSize: row col cols rows
*/
void ImageSimCtrl::setRecToImg(cv::Mat *img, int *posSize) {
    cv::Point p1(posSize[1], posSize[0]);
    cv::Point p2(posSize[1] + posSize[3], posSize[0] + posSize[2]);
    cv::rectangle(*img, p1, p2, cv::Scalar(0, 255, 0), 2);
}

/*
 * image detection algorithm for cuda
 * obj: object information
 * wkFrame: working frame information
 * judgRes: judge result which has same size of wkFrame
 * trSize: tolerance of detection method
 * threshold: detection threshold of algorithm
*/
void ImageSimCtrl::ImageDetectionCuda(
        ImgInfo *obj,
        ImgInfo *wkFrame,
        float *judgeRes,
        DeteInfo deteInfo) {

    ObjDetection(obj, wkFrame, judgeRes, deteInfo);
}


