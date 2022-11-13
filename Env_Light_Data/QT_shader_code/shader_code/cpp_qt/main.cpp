// Copyright (C) Thorsten Thormaehlen, MIT License (see license file)

#include <QApplication>
#include <QOpenGLWidget>
#include <QKeyEvent>
#include <QTimer>
#include <QMessageBox>

#include <iostream>
#include <fstream>
#include <sstream>

#include "Renderer.h"

using namespace std;
using namespace gsn;

class MyWidget : public QOpenGLWidget {

private:
  Renderer *renderer;
  QTimer *timer;

public:
  MyWidget(QWidget *parent = NULL) : QOpenGLWidget(parent) {
    this->setWindowTitle("GSN Composer Shader Export");
    this->resize(640, 640);
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(update()));
    timer->start(30);
  }

  ~MyWidget() {
    makeCurrent();
    renderer->dispose();
    doneCurrent();
    delete renderer;
  }

protected:
  void initializeGL() {
      renderer = new Renderer();
      renderer->init();
  }
  void resizeGL(int w, int h){ renderer->resize(w, h); }
  void paintGL() {
      float offset = 1.0f;
      renderer->t += offset;
      renderer->display();
  }
};

int main (int argc, char* argv[]) {
    // create a QApplication object that handles initialization,
    // finalization, and the main event loop
    QApplication appl(argc, argv);
	std::setlocale(LC_NUMERIC, "C");
    MyWidget widget;  // create a widget
    widget.show(); //show the widget and its children
    return appl.exec(); // execute the application
}
