/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.14.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionImageCalc;
    QWidget *centralwidget;
    QWidget *verticalWidget_showRes;
    QVBoxLayout *verticalLayout;
    QPushButton *pushButton_getFrameInfo;
    QLabel *label_frameNumber;
    QLineEdit *lineEdit_frameNumber;
    QPushButton *pushButton_showRes;
    QPushButton *pushButton_clear;
    QSplitter *splitter;
    QPushButton *pushButton_setWkPath;
    QLabel *label_dataSetPath;
    QPushButton *pushButton_saveImg;
    QPushButton *pushButton_setSavePath;
    QLabel *label_savePath;
    QMenuBar *menuBar;
    QMenu *menuAlgorithm;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 632);
        actionImageCalc = new QAction(MainWindow);
        actionImageCalc->setObjectName(QString::fromUtf8("actionImageCalc"));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalWidget_showRes = new QWidget(centralwidget);
        verticalWidget_showRes->setObjectName(QString::fromUtf8("verticalWidget_showRes"));
        verticalWidget_showRes->setGeometry(QRect(10, 10, 611, 481));
        verticalLayout = new QVBoxLayout(verticalWidget_showRes);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        pushButton_getFrameInfo = new QPushButton(centralwidget);
        pushButton_getFrameInfo->setObjectName(QString::fromUtf8("pushButton_getFrameInfo"));
        pushButton_getFrameInfo->setGeometry(QRect(640, 30, 101, 25));
        label_frameNumber = new QLabel(centralwidget);
        label_frameNumber->setObjectName(QString::fromUtf8("label_frameNumber"));
        label_frameNumber->setGeometry(QRect(640, 60, 101, 17));
        lineEdit_frameNumber = new QLineEdit(centralwidget);
        lineEdit_frameNumber->setObjectName(QString::fromUtf8("lineEdit_frameNumber"));
        lineEdit_frameNumber->setGeometry(QRect(640, 80, 113, 25));
        pushButton_showRes = new QPushButton(centralwidget);
        pushButton_showRes->setObjectName(QString::fromUtf8("pushButton_showRes"));
        pushButton_showRes->setGeometry(QRect(640, 140, 89, 25));
        pushButton_clear = new QPushButton(centralwidget);
        pushButton_clear->setObjectName(QString::fromUtf8("pushButton_clear"));
        pushButton_clear->setGeometry(QRect(640, 110, 89, 25));
        splitter = new QSplitter(centralwidget);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setGeometry(QRect(10, 520, 451, 25));
        splitter->setOrientation(Qt::Horizontal);
        pushButton_setWkPath = new QPushButton(splitter);
        pushButton_setWkPath->setObjectName(QString::fromUtf8("pushButton_setWkPath"));
        splitter->addWidget(pushButton_setWkPath);
        label_dataSetPath = new QLabel(splitter);
        label_dataSetPath->setObjectName(QString::fromUtf8("label_dataSetPath"));
        splitter->addWidget(label_dataSetPath);
        pushButton_saveImg = new QPushButton(centralwidget);
        pushButton_saveImg->setObjectName(QString::fromUtf8("pushButton_saveImg"));
        pushButton_saveImg->setGeometry(QRect(640, 170, 89, 25));
        pushButton_setSavePath = new QPushButton(centralwidget);
        pushButton_setSavePath->setObjectName(QString::fromUtf8("pushButton_setSavePath"));
        pushButton_setSavePath->setGeometry(QRect(10, 550, 126, 25));
        label_savePath = new QLabel(centralwidget);
        label_savePath->setObjectName(QString::fromUtf8("label_savePath"));
        label_savePath->setGeometry(QRect(140, 550, 451, 17));
        MainWindow->setCentralWidget(centralwidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 800, 22));
        menuAlgorithm = new QMenu(menuBar);
        menuAlgorithm->setObjectName(QString::fromUtf8("menuAlgorithm"));
        MainWindow->setMenuBar(menuBar);

        menuBar->addAction(menuAlgorithm->menuAction());
        menuAlgorithm->addAction(actionImageCalc);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        actionImageCalc->setText(QCoreApplication::translate("MainWindow", "ImageSimulation", nullptr));
        pushButton_getFrameInfo->setText(QCoreApplication::translate("MainWindow", "GetFrameInfo", nullptr));
        label_frameNumber->setText(QCoreApplication::translate("MainWindow", "frame number", nullptr));
        pushButton_showRes->setText(QCoreApplication::translate("MainWindow", "showResult", nullptr));
        pushButton_clear->setText(QCoreApplication::translate("MainWindow", "clear", nullptr));
        pushButton_setWkPath->setText(QCoreApplication::translate("MainWindow", "SetPath", nullptr));
        label_dataSetPath->setText(QCoreApplication::translate("MainWindow", "set dataset path", nullptr));
        pushButton_saveImg->setText(QCoreApplication::translate("MainWindow", "saveImg", nullptr));
        pushButton_setSavePath->setText(QCoreApplication::translate("MainWindow", "setImgeSavePath", nullptr));
        label_savePath->setText(QCoreApplication::translate("MainWindow", "images save path", nullptr));
        menuAlgorithm->setTitle(QCoreApplication::translate("MainWindow", "Algorithm", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
