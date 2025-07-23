#ifndef IMAGESIMCTRL_H
#define IMAGESIMCTRL_H

#include <QWidget>
#include <QFile>
#include <QDir>
#include <QtDebug>
#include <matrixBase.h>
#include <opencv2/opencv.hpp>
#include <imageprocess.h>

namespace Ui {
class ImageSimCtrl;
}

class ImageSimCtrl : public QWidget
{
    Q_OBJECT

public:
    explicit ImageSimCtrl(QWidget *parent = nullptr);
    ~ImageSimCtrl();
    void setDataPath(QString dataPath);

public:
    QString dataPath; // dataset's path

private:
    Ui::ImageSimCtrl *ui;

private:
    void startSim();
    void getLineInfo(QString fileName, float *coorRes); // get information from file(path and name)
    void getLineInfo(QString fileName, float *coorRes, int startCnt, int infoCnt);
    int getFileInfo(QString fileName); //get file information
    void getWkImg(ImgInfo *imgInfo); //get working coordinate direction vector
    void codeImg(QString imgPath, ImgInfo *imgInfo); // code image by name
    void codeImg(cv::Mat wkImg, ImgInfo *imgInfo); // code image by wkImg
    void decodeImg(cv::Mat *imgRes, ImgInfo *imgInfo); //decode image
    void decodeImg(cv::Mat *imgRes, float *deteRes, ImgInfo *imgInfo, float *decodeRes); //decode image
    void getBorder(int *wkPos, int *borderInfo); //get min max border from standard value
    void getObjTemImg(cv::Mat *img, cv::Mat *temObj, int *borderInfo); // draw rectangle in img
    void BackGrdDiffer();// start background differ algorithm
    void startImageDetection();
    void ImageDetectionCuda(ImgInfo *obj, ImgInfo *wkFrame, float *judgeRes, DeteInfo deteInfo); // image detection test for cuda
    void setRecToImg(cv::Mat *img, int *posSize); // set rectangle to img
};

#endif // IMAGESIMCTRL_H
