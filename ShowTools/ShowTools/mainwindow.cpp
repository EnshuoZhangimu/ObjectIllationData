#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->showRes = new QtDataVisualization::Q3DScatter();
    QWidget *container = QWidget::createWindowContainer(this->showRes);
    ui->verticalLayout->addWidget(container); //add showRes to mainwindow
    this->imgeSimCtrl = new ImageSimCtrl();
    this->imgeSimCtrl->hide();
    connect(ui->pushButton_setWkPath,
            &QPushButton::clicked,
            this,
            &MainWindow::setDatasetPath
            ); //set datasets path button
    connect(ui->pushButton_getFrameInfo,
            &QPushButton::clicked,
            this,
            &MainWindow::getFrameInfo); // get number of frames
    connect(ui->pushButton_showRes,
            &QPushButton::clicked,
            this,
            &MainWindow::showStandardModel); //show standard model result
    connect(ui->pushButton_clear,
            &QPushButton::clicked,
            this,
            &MainWindow::clearSeries); //clear all the series in plot
    connect(ui->pushButton_saveImg,
            &QPushButton::clicked,
            this,
            &MainWindow::saveCurrentImg); // save showRes button
    connect(ui->pushButton_setSavePath,
            &QPushButton::clicked,
            this,
            &MainWindow::setSavePath); // set showRes's save path
    connect(ui->actionImageCalc,
            &QAction::triggered,
            this,
            &MainWindow::StartImageCalcCtrl);
    this->initDataPathInfo(); // init data path information
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::initDataPathInfo() {
    QDir *dir = new QDir(QDir::currentPath() + "/dataInfo"); //datasets' info path
    if (!dir->exists()) {//if dataset path information doesn't exist
        dir->mkpath(QDir::currentPath() + "/dataInfo");
        QFile file(QDir::currentPath() + "/dataInfo/dataInfo.txt");
        file.open(QIODevice::WriteOnly);
        file.write("empty");
        file.close();
    } else {//if dataset path information exist
        QFile file(QDir::currentPath() + "/dataInfo/dataInfo.txt");
        file.open(QIODevice::ReadOnly);
        QString dataSetInfo = QString(file.readAll());
        file.close();
        if (dataSetInfo != "empty") {
            this->dataSetPath = dataSetInfo;
            ui->label_dataSetPath->setText(dataSetInfo);
        }
    }
}

void MainWindow::setDatasetPath() {
    QString dataSetPath = QFileDialog::getExistingDirectory(
                this,
                tr("select dataset"),
                this->dataSetPath
                );
    if (dataSetPath != "") {
        this->dataSetPath = dataSetPath;
        QFile file(QDir::currentPath() + "/dataInfo/dataInfo.txt");
        file.open(QIODevice::WriteOnly);
        file.write(dataSetPath.toUtf8());
        file.close();
        ui->label_dataSetPath->setText(dataSetPath);
        this->imgeSimCtrl->setDataPath(dataSetPath);
    }
}

void MainWindow::getFrameInfo() {
    QDir *dir = new QDir(this->dataSetPath);
    QStringList filter; //frame name filter
    filter << "frame*";
    QStringList frameArray = dir->entryList(filter, QDir::Dirs);
//    qDebug() << frameArray.length();
    ui->label_frameNumber->setText(QString::number(frameArray.length()) + " frames");
}

void MainWindow::showStandardModel() {
    this->clearSeries();
    QString frameName = "/frame" + ui->lineEdit_frameNumber->text();
//    qDebug() << frameName;
    QDir *dir = new QDir(this->dataSetPath + frameName + "/StdObj");
    QStringList filter;
    filter << "HG*";
    QStringList objNameArray = dir->entryList(filter, QDir::Dirs);
//    qDebug() << objNameArray;
    foreach (QString objName, objNameArray) {
        dir->setPath(this->dataSetPath + frameName + "/StdObj/" + objName);
        QStringList partNameArray = dir->entryList(QDir::Files); // get all parts' name array
        foreach (QString partName, partNameArray) {
//            qDebug() << partName;
            this->showSingleObj(this->dataSetPath + frameName + "/StdObj/" + objName + "/" + partName);
        }
    }
    this->showRes->scene()->activeCamera()->setCameraPreset(QtDataVisualization::Q3DCamera::CameraPresetBehindBelow);
    this->calcShowScale();
    /*set the show scale of all data*/
    this->showRes->axisX()->setMin(this->showScale[0]);
    this->showRes->axisX()->setMax(this->showScale[1]);
    this->showRes->axisY()->setMin(this->showScale[2]);
    this->showRes->axisY()->setMax(this->showScale[3]);
    this->showRes->axisZ()->setMin(this->showScale[4]);
    this->showRes->axisZ()->setMax(this->showScale[5]);
    this->showScale.clear();
    /*reset the show scale*/
    this->isFirstCheck = true;
    this->showRes->show();
//    this->saveCurrentImg();
}

void MainWindow::showSingleObj(QString pointFile) {
    QFile file(pointFile);
    file.open(QIODevice::ReadOnly);
    QString pointCloud = QString(file.readAll()); // get point cloud information
    int startIndex = 0; // start index of point cloud information
    int contentLength = 0; //content length of a information
    QStringList pointInfos; // point cloud information in list formate
    for(int i = 0;i < pointCloud.length();i++) {
        if (pointCloud[i] == "\n") {
            pointInfos << pointCloud.mid(startIndex, contentLength); // when \n appears, info is end
            startIndex = i + 1; // next info start index
            contentLength = 0; // reset the contentlenght
        } else {
            contentLength += 1;
        }
    }
//    qDebug() << pointInfos;
    QtDataVisualization::QScatterDataArray data;
    foreach (QString pointInfo, pointInfos) {
        std::vector<int> SpaceIndices;
        for (int i = 0; i < pointInfo.length(); ++i) {
            if (pointInfo[i] == " ") {
                SpaceIndices.push_back(i);
            }
        }
        std::vector<float> singleCoor; // (x, y, z)
        for (int i = 0; i < 2; ++i) {
            QString addCache = pointInfo.mid(SpaceIndices[i] + 1, SpaceIndices[i + 1] - SpaceIndices[i] - 1);
//            qDebug() << addCache;
            singleCoor.push_back(addCache.toFloat());
        }
        singleCoor.push_back(pointInfo.mid(SpaceIndices[2] + 1).toFloat());
//        qDebug() << singleCoor;
        QVector3D pointCache;
        pointCache.setX(singleCoor[0]);
        pointCache.setY(singleCoor[1]);
        pointCache.setZ(singleCoor[2]);
        data << pointCache;
        this->calcShowCube(singleCoor);
    }
    this->showRes->setFlags(this->showRes->flags() ^ Qt::FramelessWindowHint);
    QtDataVisualization::QScatter3DSeries *series = new QtDataVisualization::QScatter3DSeries;
    series->dataProxy()->addItems(data);
    series->setItemSize(0.01);
    series->setMesh(QtDataVisualization::QAbstract3DSeries::MeshPoint);
    this->showRes->addSeries(series);
    this->showRes->setShadowQuality(this->showRes->ShadowQualityNone);
}

void MainWindow::clearSeries() {
    QList<QtDataVisualization::QScatter3DSeries *> seriesList = this->showRes->seriesList();
    foreach (QtDataVisualization::QScatter3DSeries *series, seriesList) {
        this->showRes->removeSeries(series);
    }
}

/*
 * calculate show cube of frame
 * singleCoor: coordinate of single point frame objects' point cloud
*/
void MainWindow::calcShowCube(std::vector<float> singleCoor) {
    if (this->isFirstCheck) {// set the first point as default value
        for (int i = 0;i < 3;i++) {
            this->showScale.push_back(singleCoor[i]);
            this->showScale.push_back(singleCoor[i]);
        }
        this->isFirstCheck = false;
    } else {
        for (int i = 0; i < 3; ++i) {
            if (this->showScale[2*i] > singleCoor[i]) {
                this->showScale[2*i] = singleCoor[i];
            }
            if (this->showScale[2*i + 1] < singleCoor[i]) {
                this->showScale[2*i + 1] = singleCoor[i];
            }
        }
    }
}

void MainWindow::calcShowScale() {
    std::vector<float> disArray; // distance array (x y z)
    std::vector<float> centerArray; // center array of x y z
    /*calculate all the distance of coordinate*/
    for (int i = 0; i < 3; ++i) {
        disArray.push_back(this->showScale[2*i + 1] - this->showScale[2*i]);
        centerArray.push_back((this->showScale[2*i + 1] + this->showScale[2*i])/2);
    }
    /*find the max distance of coordinate*/
    float maxDis = disArray[0]; // set the 1st value as default
    for (int i = 1; i < 3; i++) {
        if (maxDis < disArray[i]) {
            maxDis = disArray[i];
        }
    }
    maxDis = maxDis/2;
    for (int i = 1; i < 3; i++) {
        this->showScale[2*i] = centerArray[i] - maxDis;
        this->showScale[2*i + 1] = centerArray[i] + maxDis;
    }
}

void MainWindow::saveCurrentImg() {
    QImage img = this->showRes->renderToImage();
    img.save(this->imgSavePath + "/state.png");
}

void MainWindow::setSavePath() {
    this->imgSavePath = QFileDialog::getExistingDirectory(
                this,
                tr("open directory"),
                QDir::currentPath()
                );
    ui->label_savePath->setText(this->imgSavePath);
}

void MainWindow::StartImageCalcCtrl() {
//    qDebug() << "start Image Ctrl Panel";
    this->imgeSimCtrl->show();
}














