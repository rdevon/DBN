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

#define GLFW_GL3
#include <GL/glfw.h>
#define OPENGL3
#include "opengl.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


class Viz_Unit;
class GL_Unit;
class Learning_Monitor;
class Unit_Monitor;
class Layer;
class Connection;
class Pathway;
class Connection_Weight_Monitor;
class Layer_Sample_Monitor;
class Layer_Bias_Monitor;
class Reconstruction_Cost_Monitor;



class Visualizer{
public:
   
   std::vector<GLuint> _texID;
   std::vector<GLuint>  _lineVAO;   //< Vertex array object for line plots
   std::vector<GLuint>  _lineBO;    //< Vertex buffer object for the position of vertices in the line plots
   
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
   GLint  _weightSamplerLoc;    //< Location of the weight texture sampler in the fragment program
   std::vector<glm::vec4> _points;   //< List of points for the textured quad
   std::vector<glm::vec2> _texCoords;//< Texture coordinates for the textured quad
   
   Visualizer(){}
   Visualizer(int width, int height){
      open_window(width, height);
   }
   
   void open_window(int width, int height);
   void close_window();
   void update();
   
   //GL STUFF HERE
   
   void terminate(int exitCode);
   std::string readTextFile(const std::string& filename);
   bool shaderCompileStatus(GLuint shader);
   std::string getShaderLog(GLuint shader);
   bool programLinkStatus(GLuint program);
   std::string getProgramLog(GLuint program);
   GLuint createShader(const std::string& source, GLenum shaderType);
   GLuint createGLSLProgram();
   void init(int num_tex_maps, int num_line_plots);
   int update_line_units(Unit_Monitor *gl_unit, int id);
   int update_tex_unit(Unit_Monitor *gl_unit, int id);
   int update(Learning_Monitor*);
   
};

class Learning_Monitor {
public:
   Visualizer                         *viz;
   
   std::map <int, Unit_Monitor*>      tex_units;
   std::map <int, Unit_Monitor*>      line_units;
   
   Learning_Monitor(){viz = new Visualizer(1000,1000);}
   void update();
};

class Connection_Learning_Monitor : public Learning_Monitor {
public:
   enum TexMaps {
      BOT_SAMPLES = 0,
      CONNECTION_WEIGHTS,
      TOP_SAMPLES,
      TOP_BIASES,
      NUM_TEX_MAPS
   };
   enum LinePlots {
      REC_COST_PLOT,
      NUM_LINE_PLOTS
   };
      
   Layer_Sample_Monitor          *top_sample_monitor;
   Layer_Sample_Monitor          *bot_sample_monitor;
   Connection_Weight_Monitor     *con_weight_monitor;
   
   Layer_Bias_Monitor            *top_bias_monitor;
   
   Reconstruction_Cost_Monitor   *rec_cost_monitor;
   
   Connection_Learning_Monitor(Connection* connection);
   
};

class Monitor_Unit {
public:
   
   int                     image_pixels_x,
                           image_pixels_y;
   
   float                   value;
   
   Monitor_Unit(){}
};

class Unit_Monitor {
public:
   
   std::string             name;
   
   // GL positions
   float                   x_position,
                           y_position,
                           z_position;
   int                     x_size,
                           y_size,
                           z_size;
   
   float                   color[4];
   
   float                   threshold;
   
   Monitor_Unit            *unit;
   
   gsl_vector_float        *viz_vector;
   gsl_matrix_float        *viz_matrix;
   std::vector<float>      plot_vector;
   
   int                     pieces_across,
                           pieces_down,
                           piece_count;
   
   Unit_Monitor(){}
   
   void add_viz_vector();
   void clear_viz();
   virtual void load_viz();
   void finish_setup();
   void plot();
   void set_coords(float x, float y, float z){x_position = x, y_position = y, z_position = z;}
   void scale_matrix_and_threshold();
};

class Layer_Bias_Monitor : public Unit_Monitor {
public:
   Layer_Bias_Monitor(Layer* layer, int x_pixels = 200);
   void load_viz();
};

class Layer_Sample_Monitor : public Unit_Monitor {
public:
   Layer_Sample_Monitor(Layer* layer, int x_pixels = 200, int y_pixels = 100, float thresh = .1);
   void load_viz();
};

class Connection_Weight_Monitor : public Unit_Monitor {
public:
   int sample_number;
   Connection_Weight_Monitor(Connection* connection, int sample_num, int x_pixels = 500, int y_pixels = 200, float thresh = .1);
   void load_viz();
};
/*
class Pathway_Monitor : public Unit_Monitor {
public:
   Pathway_Monitor(Pathway* pathway, int pieces_across, int pieces_down, float scale = 1, float value_scale = 1, float threshold = 1);
   void load_viz();
};*/

class Reconstruction_Cost_Monitor : public Unit_Monitor {
public:
   Reconstruction_Cost_Monitor(Layer* layer);
   void load_viz();
};

void GLFWCALL resize(int width, int height);
void GLFWCALL keypress(int key, int state);
int GLFWCALL close(void);

#endif