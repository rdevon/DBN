//
//  Viz.h
//  DBN
//
//  Created by Devon Hjelm on 7/23/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_Viz_h
#define DBN_Viz_h

#include <stdlib.h>
#include "Params.h"

#ifdef USEGL

#if __APPLE__
#define OPENGL3
#endif
#include "opengl.h"
#define GLFW_GL3
#include <GL/glfw.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#endif

class Tex_Unit;
class Plot_Unit;
class Unit_Monitor;
class Feature_to_Data_Monitor;
class Monitor;
class New_Monitor;
class MLP;
class Layer;
class fMRI_Feature_Monitor;
class Border;
class fMRI_Layer_Monitor;
class Reconstruction_Cost_Monitor;
class DBN;
class MNIST_Feature_Monitor;
class MNIST_Layer_Monitor;
class Teacher;
class Visualizer;
class Viz_Unit;

#ifdef USEGL
void GLFWCALL resize(int width, int height);
void GLFWCALL keypress(int key, int state);
int GLFWCALL close(void);
extern Visualizer *the_viz;
#endif

extern Monitor    *the_monitor;

class Visualizer{
public:
   
   bool                    on;
   bool                    pause;
   enum tex_control{POSITVE = 0, NEGATIVE, BOTH} tc;
#ifdef USEGL
   std::map<Tex_Unit*, GLuint> _texture_maps;
   std::map<Plot_Unit*, GLuint> _lineVAO_maps;
   std::map<Plot_Unit*, GLuint> _lineBO_maps;
   std::map<Viz_Unit*, GLuint> _borderVAO_maps;
   std::map<Viz_Unit*, GLuint> _borderBO_maps;
   
   //std::map<GLuint, Tex_Unit*>  _texID;
   //std::map<GLuint, Plot_Unit*>  _lineVAO;   //< Vertex array object for line plots
   //std::map<GLuint, Plot_Unit*>  _lineBO;    //< Vertex buffer object for the position of vertices in the line plots
   
   GLuint _program;             //< Shader program handle
   GLuint _vao;                 //< Vertex array object for the vertices for the textured quad
   GLuint _vertexBuf;           //< Buffer object for the vertices for the textured quad
   GLuint _texCoordBuf;         //< Buffer object for the texture coordinates for the textured quad
   GLint  _vertexLocation;      //< Location of the vertex attribute in the shader program
   GLint  _colorLocation;       //< Location of color!!!
   GLint  _scaleLocation;       //< Location of scale!!!
   GLint  _texCoordLocation;    //< Location of the texture coordinate attribute in the shader program
   GLint  _mvpLocation;         //< model view projection matrix.  This is how we'll move things around in the window
   GLuint _texWeightId;         //< Texture object for the weights
   GLuint _negtexWeightId;
   GLuint _plotId;
   GLuint _colorFilter;
   bool   _linearFilter;        //< TODO
   GLint  _weightSamplerLoc;    //< Location of the weight texture sampler in the fragment program
   std::vector<glm::vec4> _points;   //< List of points for the textured quad
   std::vector<glm::vec2> _texCoords;//< Texture coordinates for the textured quad
#endif
   Visualizer(){}
#ifdef USEGL
   Visualizer(int width, int height){
      open_window(width, height);
      pause = false;
      tc = POSITVE;
   }
   
   void open_window(int width, int height);
   void close_window();
   void add_tex(Tex_Unit*);
   void add_plot(Plot_Unit*);
   void delete_tex(Tex_Unit*);
   void delete_plot(Plot_Unit*);
   void clear();
   
   void terminate(int exitCode);
   std::string readTextFile(const std::string& filename);
   bool shaderCompileStatus(GLuint shader);
   std::string getShaderLog(GLuint shader);
   bool programLinkStatus(GLuint program);
   std::string getProgramLog(GLuint program);
   GLuint createShader(const std::string& source, GLenum shaderType);
   GLuint createGLSLProgram();
   
   void init();
   
   int draw_texture_map(Tex_Unit *tex_unit);
   int draw_plot(Plot_Unit *plot_unit);
   int draw_border(Viz_Unit *viz_unit);
   int update(Monitor*);
   void toggle_on() {on = !on;}
   void toggle_pause() {pause = !pause;
      if (pause) std::cout << "<PAUSE>" << std::endl; else std::cout << "<RESUME>" << std::endl; 
   }
#endif
};
#endif
