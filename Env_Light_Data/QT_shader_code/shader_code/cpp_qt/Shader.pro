QT += widgets opengl openglwidgets
TARGET = Shader
HEADERS += FileTools.h LoadOBJ.h \
		   Matrix.h  Mesh.h \
		   Renderer.h ShaderNode.h \
		   StringTools.h
SOURCES += main.cpp \
           FileTools.cpp LoadOBJ.cpp\
           Matrix.cpp Mesh.cpp\
		   Renderer.cpp ShaderNode.cpp \
		   StringTools.cpp
CONFIG += console