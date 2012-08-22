//
//  Viz.h
//  DBN
//
//  Created by Devon Hjelm on 7/23/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_Viz_h
#define DBN_Viz_h
#include "Types.h"
#include "Layers.h"
#include "IO.h"

#define GLFW_GL3
#include <GL/glfw.h>
#define OPENGL3
#include "opengl.h"

#include <glm/glm.hpp>

class Visualizer{
public:
   
   std::string name;
   float thresh_;
   int count, across, down;
   int imageW, imageH; 
   gsl_matrix_float *viz;
   
   DataSet *data_;
   
   Visualizer(){}
   Visualizer(int minsamples, DataSet *data, std::string newname = ""): count(0) {
      data_ = data;
      if (newname == "") name = data->name;
      else name = newname;
      int pixelWidth = 250;
      //number of samples across and down
      across = pixelWidth/(data->width);
      down = ceilf((float)minsamples/(float)across);
      //across = 3;
      //down = 6;
      imageH = data->height;
      imageW = data->width;
      viz = gsl_matrix_float_calloc(imageH*down, imageW*across);
      initViz();
   }
   
   void add(gsl_vector_float *sample);
   void clear();
   void plot();
   
   void initViz();
   void updateViz();
   void close_window();
};

//GL STUFF HERE

void terminate(int exitCode);
std::string readTextFile(const std::string& filename);
bool shaderCompileStatus(GLuint shader);
std::string getShaderLog(GLuint shader);
bool programLinkStatus(GLuint program);
std::string getProgramLog(GLuint program);
GLuint createShader(const std::string& source, GLenum shaderType);
GLuint createGLSLProgram();
void init(void);
void GLFWCALL resize(int width, int height);
void GLFWCALL keypress(int key, int state);
int GLFWCALL close(void);
int update(double time);

#endif
