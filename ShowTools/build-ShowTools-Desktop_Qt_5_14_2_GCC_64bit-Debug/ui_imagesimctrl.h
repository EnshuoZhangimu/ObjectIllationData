/********************************************************************************
** Form generated from reading UI file 'imagesimctrl.ui'
**
** Created by: Qt User Interface Compiler version 5.14.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_IMAGESIMCTRL_H
#define UI_IMAGESIMCTRL_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ImageSimCtrl
{
public:
    QLabel *label_DataPath;
    QPushButton *pushButton_startSim;
    QSplitter *splitter;
    QLabel *label_pixelDiff;
    QLineEdit *lineEdit_pixelDiff;
    QSplitter *splitter_3;
    QLabel *label_bloclCnt;
    QLineEdit *lineEdit_blockCnt;
    QSplitter *splitter_2;
    QLabel *label_perThread;
    QLineEdit *lineEdit_perThread;
    QPushButton *pushButton_Differ;
    QPushButton *pushButton_DeteSim;
    QSplitter *splitter_4;
    QLabel *label_trSize;
    QLineEdit *lineEdit_trSize;
    QSplitter *splitter_5;
    QLabel *label_threshold;
    QLineEdit *lineEdit_threshold;

    void setupUi(QWidget *ImageSimCtrl)
    {
        if (ImageSimCtrl->objectName().isEmpty())
            ImageSimCtrl->setObjectName(QString::fromUtf8("ImageSimCtrl"));
        ImageSimCtrl->resize(775, 615);
        label_DataPath = new QLabel(ImageSimCtrl);
        label_DataPath->setObjectName(QString::fromUtf8("label_DataPath"));
        label_DataPath->setGeometry(QRect(50, 70, 411, 17));
        pushButton_startSim = new QPushButton(ImageSimCtrl);
        pushButton_startSim->setObjectName(QString::fromUtf8("pushButton_startSim"));
        pushButton_startSim->setGeometry(QRect(50, 150, 89, 25));
        splitter = new QSplitter(ImageSimCtrl);
        splitter->setObjectName(QString::fromUtf8("splitter"));
        splitter->setGeometry(QRect(50, 250, 213, 22));
        splitter->setOrientation(Qt::Horizontal);
        label_pixelDiff = new QLabel(splitter);
        label_pixelDiff->setObjectName(QString::fromUtf8("label_pixelDiff"));
        splitter->addWidget(label_pixelDiff);
        lineEdit_pixelDiff = new QLineEdit(splitter);
        lineEdit_pixelDiff->setObjectName(QString::fromUtf8("lineEdit_pixelDiff"));
        splitter->addWidget(lineEdit_pixelDiff);
        splitter_3 = new QSplitter(ImageSimCtrl);
        splitter_3->setObjectName(QString::fromUtf8("splitter_3"));
        splitter_3->setGeometry(QRect(50, 96, 213, 23));
        splitter_3->setOrientation(Qt::Horizontal);
        label_bloclCnt = new QLabel(splitter_3);
        label_bloclCnt->setObjectName(QString::fromUtf8("label_bloclCnt"));
        splitter_3->addWidget(label_bloclCnt);
        lineEdit_blockCnt = new QLineEdit(splitter_3);
        lineEdit_blockCnt->setObjectName(QString::fromUtf8("lineEdit_blockCnt"));
        splitter_3->addWidget(lineEdit_blockCnt);
        splitter_2 = new QSplitter(ImageSimCtrl);
        splitter_2->setObjectName(QString::fromUtf8("splitter_2"));
        splitter_2->setGeometry(QRect(50, 123, 213, 22));
        splitter_2->setOrientation(Qt::Horizontal);
        label_perThread = new QLabel(splitter_2);
        label_perThread->setObjectName(QString::fromUtf8("label_perThread"));
        splitter_2->addWidget(label_perThread);
        lineEdit_perThread = new QLineEdit(splitter_2);
        lineEdit_perThread->setObjectName(QString::fromUtf8("lineEdit_perThread"));
        splitter_2->addWidget(lineEdit_perThread);
        pushButton_Differ = new QPushButton(ImageSimCtrl);
        pushButton_Differ->setObjectName(QString::fromUtf8("pushButton_Differ"));
        pushButton_Differ->setGeometry(QRect(50, 280, 89, 25));
        pushButton_DeteSim = new QPushButton(ImageSimCtrl);
        pushButton_DeteSim->setObjectName(QString::fromUtf8("pushButton_DeteSim"));
        pushButton_DeteSim->setGeometry(QRect(50, 180, 191, 25));
        splitter_4 = new QSplitter(ImageSimCtrl);
        splitter_4->setObjectName(QString::fromUtf8("splitter_4"));
        splitter_4->setGeometry(QRect(280, 100, 225, 23));
        splitter_4->setOrientation(Qt::Horizontal);
        label_trSize = new QLabel(splitter_4);
        label_trSize->setObjectName(QString::fromUtf8("label_trSize"));
        splitter_4->addWidget(label_trSize);
        lineEdit_trSize = new QLineEdit(splitter_4);
        lineEdit_trSize->setObjectName(QString::fromUtf8("lineEdit_trSize"));
        splitter_4->addWidget(lineEdit_trSize);
        splitter_5 = new QSplitter(ImageSimCtrl);
        splitter_5->setObjectName(QString::fromUtf8("splitter_5"));
        splitter_5->setGeometry(QRect(280, 127, 225, 23));
        splitter_5->setOrientation(Qt::Horizontal);
        label_threshold = new QLabel(splitter_5);
        label_threshold->setObjectName(QString::fromUtf8("label_threshold"));
        splitter_5->addWidget(label_threshold);
        lineEdit_threshold = new QLineEdit(splitter_5);
        lineEdit_threshold->setObjectName(QString::fromUtf8("lineEdit_threshold"));
        splitter_5->addWidget(lineEdit_threshold);

        retranslateUi(ImageSimCtrl);

        QMetaObject::connectSlotsByName(ImageSimCtrl);
    } // setupUi

    void retranslateUi(QWidget *ImageSimCtrl)
    {
        ImageSimCtrl->setWindowTitle(QCoreApplication::translate("ImageSimCtrl", "ImageSimCtrl", nullptr));
        label_DataPath->setText(QCoreApplication::translate("ImageSimCtrl", "DataPath", nullptr));
        pushButton_startSim->setText(QCoreApplication::translate("ImageSimCtrl", "StartImgSim", nullptr));
        label_pixelDiff->setText(QCoreApplication::translate("ImageSimCtrl", "pixelDiff", nullptr));
        label_bloclCnt->setText(QCoreApplication::translate("ImageSimCtrl", "blockCnt", nullptr));
        label_perThread->setText(QCoreApplication::translate("ImageSimCtrl", "perThread", nullptr));
        pushButton_Differ->setText(QCoreApplication::translate("ImageSimCtrl", "startDiffer", nullptr));
        pushButton_DeteSim->setText(QCoreApplication::translate("ImageSimCtrl", "StartObjDetetionAlgorithm", nullptr));
        label_trSize->setText(QCoreApplication::translate("ImageSimCtrl", "tolerantSize", nullptr));
        label_threshold->setText(QCoreApplication::translate("ImageSimCtrl", "threshold", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ImageSimCtrl: public Ui_ImageSimCtrl {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_IMAGESIMCTRL_H
