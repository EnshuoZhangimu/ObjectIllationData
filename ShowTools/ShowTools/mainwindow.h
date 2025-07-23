#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QtDataVisualization>
#include <QImage>
#include <imagesimctrl.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

signals:
    void changePath(QString path);

private://private parameters
    Ui::MainWindow *ui;
    QtDataVisualization::Q3DScatter *showRes; // show the objects' point cloud
    QString dataSetPath; //dataset's path
    std::vector<float> showScale; // plot scale [xmin, xmax, ymin, ymax, zmin, zmax]
    bool isFirstCheck = true; // check whether the point is first check point
    QString imgSavePath; // images' save path(graphics image from showRes)
    ImageSimCtrl *imgeSimCtrl; // image simulation algorithm control panel

private://private methods
    void initDataPathInfo(); //init dataset's path information
    void setDatasetPath(); //set dataset path
    void getFrameInfo(); //get frame information (number of frames)
    void showStandardModel(); //show standard human model
    void showSingleObj(QString pointFile); //show single object
    void clearSeries(); //clear series
    void calcShowCube(std::vector<float> singleCoor); //calculate showcube of frame
    void calcShowScale(); //calculate show scale of frame
    void saveCurrentImg(); // save current image of showRes
    void setSavePath(); // set showRes's save path
    void StartImageCalcCtrl(); // start image calculate control panel
};
#endif // MAINWINDOW_H
