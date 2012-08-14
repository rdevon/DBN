//
//  Viz.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/24/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "Viz.h"

//BEGIN OPENGL stuff

using namespace glm;

GLuint _program;             //< Shader program handle
GLuint _vao;                 //< Vertex array object for the vertices
GLuint _vertexBuf;           //< Buffer object for the vertices
GLuint _texCoordBuf;         //< Buffer object for the texture coordinates
GLint  _vertexLocation;      //< Location of the vertex attribute in the shader program
GLint  _texCoordLocation;    //< Location of the texture coordinate attribute in the shader program
GLuint _texWeightId;         //< Texture object for the weights
bool   _running;             //< true if the program is running, false if it is time to terminate
GLint  _weightSamplerLoc;    //< Location of the weight texture sampler in the fragment program
std::vector<vec4> _points;   //< List of points for the textured quad
std::vector<vec2> _texCoords;//< Texture coordinates for the textured quad
const char* _vertexSource =
"#version 150\n"
"\n"
"in vec4 vertex;\n"
"in vec2 texCoord;\n"
"out vec2 tc;\n"
"\n"
"void main(void)\n"
"{\n"
"	gl_Position = vertex;\n"
"  tc = texCoord;\n"
"}\n";

const char* _fragmentSource =
"#version 150\n"
"\n"
"uniform sampler2D weight;\n"
"out vec4 fragColor;\n"
"in vec2 tc;\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"\n"
"void main(void)\n"
"{\n"
"	fragColor = vec4(1.0, 1.0, 0.0, 1.0);\n"
"	fragColor = vec4(tc, 0, 1);\n"
"  fragColor = texture(weight, tc);\n"
"}\n";

/**
 * Clean up and exit
 *
 * @param exitCode      The exit code, eg, EXIT_SUCCESS or EXIT_FAILURE
 */
void terminate(int exitCode)
{
   // Delete vertex buffer object
   if(_vertexBuf)
   {
      glDeleteBuffers(1, &_vertexBuf);
      _vertexBuf = 0;
   }
   
   // Delete vertex buffer object
   if(_texCoordBuf)
   {
      glDeleteBuffers(1, &_texCoordBuf);
      _texCoordBuf = 0;
   }
   
   // Delete vertex array object
   if(_vao)
   {
      glDeleteVertexArrays(1, &_vao);
   }
   
   // Delete shader program
   if(_program)
   {
      glDeleteProgram(_program);
   }
   
   glfwTerminate();
   
   exit(exitCode);
}

/**
 * Creates a string by reading a text file.
 *
 * @param filename	The name of the file
 * @return			A string that contains the contents of the file
 */
std::string readTextFile(const std::string& filename)
{
   std::ifstream infile(filename.c_str()); // File stream
   std::string source;                     // Text file string
   std::string line;                       // A line in the file
   
   // Make sure the file could be opened
   if(!infile.is_open())
   {
      std::cerr << "Could not open file: " << filename << std::endl;
      terminate(EXIT_FAILURE);
   }
   
   // Read in the source one line at a time, then append it
   // to the source string. Not efficient.
   while(infile.good())
   {
      getline(infile, line);
      source = source + line + "\n";
   }
   
   infile.close();
   return source;
}

/**
 * Check the compile status of a shader
 *
 * @param shader     Handle to a shader
 * @return           true if the shader was compiled, false otherwise
 */
bool shaderCompileStatus(GLuint shader)
{
   GLint compiled;
   glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
   return compiled ? true : false;
}

/**
 * Retrieve a shader log
 *
 * @param shader     Handle to a shader
 * @return           The contents of the log
 */
std::string getShaderLog(GLuint shader)
{
   // Get the size of the log and allocate the required space
   GLint size;
   glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &size);
   
   // Allocate space for the log
   char* log = new char[size];
   
   // Get the shader log
   glGetShaderInfoLog(shader, size, NULL, log);
   
   // Convert it into a string (not efficient)
   std::string retval(log);
   
   // Free up space
   delete [] log;
   
   return retval;
}

/**
 * Check the link status of a program
 *
 * @param shader     Handle to a shader
 * @return           true if the shader was compiled, false otherwise
 */
bool programLinkStatus(GLuint program)
{
   GLint linked;
   glGetProgramiv(_program, GL_LINK_STATUS, &linked);
   return linked ? true : false;
}

/**
 * Retrieve a GLSL program log
 *
 * @param shader     Handle to a GLSL program
 * @return           The contents of the log
 */
std::string getProgramLog(GLuint program)
{
   // Get the size of the log and allocate the required space
   GLint size;
   glGetProgramiv(_program, GL_INFO_LOG_LENGTH, &size);
   char* log = new char[size];
   
   // Get the program log
   glGetProgramInfoLog(program, size, NULL, log);
   
   // Convert it into a string (blah)
   std::string retval(log);
   
   // Clean up and return
   delete [] log;
   return retval;
}

/**
 * Create a shader program from a source string. The caller should
 * check the compile status
 *
 * @param source        The shader source
 * @param shaderType    The type of shader (GL_VERTEX_SHADER, etc)
 * @return              A handle to the shader program. 0 if an
 *                      error occured.
 */
GLuint createShader(const std::string& source, GLenum shaderType)
{
   GLuint shader = glCreateShader(shaderType);
   
   const GLchar* sourcePtr0 = source.c_str();
   const GLchar** sourcePtr = &sourcePtr0;
   
   // Set the source and attempt compilation
   glShaderSource(shader, 1, sourcePtr, NULL);
   glCompileShader(shader);
   
   return shader;
}

/**
 * Create a GLSL program object from vertex and fragment shader files.
 *
 * @param  vShaderFile   The vertex shader filename
 * @param  fShaderFile   The fragment shader filename
 * @return handle to the GLSL program
 */
GLuint createGLSLProgram()
{
   std::string vertexSource(_vertexSource);
   std::string fragmentSource(_fragmentSource);
   
   _program = glCreateProgram();
   
   // Create vertex shader
   GLuint vertexShader  = createShader(vertexSource, GL_VERTEX_SHADER);
   
   // Check for compile errors
   if(!shaderCompileStatus(vertexShader))
   {
      std::cerr << "Could not compile " << vertexShader << std::endl;
      std::cerr << getShaderLog(vertexShader) << std::endl;
      terminate(EXIT_FAILURE);
   }
   
   // Create fragment shader
   GLuint fragmentShader = createShader(fragmentSource, GL_FRAGMENT_SHADER);
   
   // Check for compile errors
   if(!shaderCompileStatus(fragmentShader))
   {
      std::cerr << "Could not compile " << vertexShader << std::endl;
      std::cerr << getShaderLog(fragmentShader) << std::endl;
      terminate(EXIT_FAILURE);
   }
   
   // Attach the shaders to the program
   glAttachShader(_program, vertexShader);
   glAttachShader(_program, fragmentShader);
   
   // Link the program
   glLinkProgram(_program);
   
   // Check for linker errors
   if(!programLinkStatus(_program))
   {
      std::cerr << "GLSL program failed to link:" << std::endl;
      std::cerr << getProgramLog(_program) << std::endl;
      terminate(EXIT_FAILURE);
   }
   
   return _program;
}

/**
 * Initialize vertex array objects, vertex buffer objects,
 * clear color and depth clear value
 */
void init(void)
{
#ifndef __APPLE__
   // GLEW has trouble supporting the core profile
   glewExperimental = GL_TRUE;
   glewInit();
   if(!GLEW_ARB_vertex_array_object)
   {
      std::cerr << "ARB_vertex_array_object not available." << std::endl;
      terminate(EXIT_FAILURE);
   }
#endif
   // Vertices for the textured quad
   _points.push_back(vec4(-1, -1, 0, 1));
   _points.push_back(vec4(-1,  1, 0, 1));
   _points.push_back(vec4( 1, -1, 0, 1));
   _points.push_back(vec4( 1,  1, 0, 1));
   
   // Texture coordinates for the textured quad
   _texCoords.push_back(vec2(0,1));
   _texCoords.push_back(vec2(0,0));
   _texCoords.push_back(vec2(1,1));
   _texCoords.push_back(vec2(1,0));
   
   // Create the shader program
   _program = createGLSLProgram();
   // Use the shader program that was loaded, compiled and linked
   glUseProgram(_program);
   
   
   // Generate a single handle for a vertex array
   glGenVertexArrays(1, &_vao);
   
   // Bind that vertex array
   glBindVertexArray(_vao);
   
   // Get the location of the "vertex" attribute in the shader program
   _vertexLocation = glGetAttribLocation(_program, "vertex");
   _texCoordLocation = glGetAttribLocation(_program, "texCoord");
   _weightSamplerLoc = glGetUniformLocation(_program, "weight");
   
   // Generate one handle for the vertex buffer object
   glGenBuffers(1, &_vertexBuf);
   
   // Make that vbo the current array buffer. Subsequent array buffer operations
   // will affect this vbo
   //
   // It is possible to place all data into a single buffer object and use
   // offsets to tell OpenGL where the data for a vertex array or any other
   // attribute may reside.
   glBindBuffer(GL_ARRAY_BUFFER, _vertexBuf);
   
   // Set the data for the vbo. This will load it onto the GPU
   glBufferData(GL_ARRAY_BUFFER,                // Target buffer object
                sizeof(vec4) * _points.size(),  // Size in bytes of the buffer
                (GLfloat*) &_points[0],         // Pointer to the data
                GL_STATIC_DRAW);                // Expected data usage pattern
   
   // Specify the location and data format of the array of generic vertex attributes
   glVertexAttribPointer(_vertexLocation, // Attribute location in the shader program
                         4,               // Number of components per attribute
                         GL_FLOAT,        // Data type of attribute
                         GL_FALSE,        // GL_TRUE: values are normalized or
                         // GL_FALSE: values are converted to fixed point
                         0,               // Stride
                         0);              // Offset into VBO for this data
   
   // Enable the generic vertex attribute array
   glEnableVertexAttribArray(_vertexLocation);
   
   glGenBuffers(1, &_texCoordBuf);
   glBindBuffer(GL_ARRAY_BUFFER, _texCoordBuf);
   // Set the data for the vbo. This will load it onto the GPU
   glBufferData(GL_ARRAY_BUFFER,                // Target buffer object
                sizeof(vec2) * _texCoords.size(),  // Size in bytes of the buffer
                (GLfloat*) &_texCoords[0],         // Pointer to the data
                GL_STATIC_DRAW);                // Expected data usage pattern
   
   // Specify the location and data format of the array of generic vertex attributes
   glVertexAttribPointer(_texCoordLocation, // Attribute location in the shader program
                         2,               // Number of components per attribute
                         GL_FLOAT,        // Data type of attribute
                         GL_FALSE,        // GL_TRUE: values are normalized or
                         // GL_FALSE: values are converted to fixed point
                         0,               // Stride
                         0);              // Offset into VBO for this data
   
   // Enable the generic vertex attribute array
   glEnableVertexAttribArray(_texCoordLocation);
   
   // Set up texture object
   
   GLsizei _texWidth = 256;
   GLsizei _texHeight = 256;
   
   GLfloat texels[_texWidth][_texHeight][4];
   
   // Create a checkerboard pattern
   for(int i = 0; i < _texWidth; i++ )
   {
      for(int j = 0; j < _texHeight; j++ )
      {
         GLubyte c = (((i & 0x8) == 0) ^ ((j & 0x8)  == 0)) * 255;
         texels[i][j][0] = c / (255.0f * 1.5f);
         texels[i][j][1] = 0;
         texels[i][j][2] = c / 255.0f;
         texels[i][j][3] = 1.0f;
      }
   }
   glGenTextures(1, &_texWeightId);
   glBindTexture(GL_TEXTURE_2D, _texWeightId);
   
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _texWidth, _texHeight, 0, GL_RGBA, GL_FLOAT, texels);
   GL_ERR_CHECK();
   glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
   GL_ERR_CHECK();
   glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
   GL_ERR_CHECK();
   glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
   GL_ERR_CHECK();
   glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
   GL_ERR_CHECK();
   glActiveTexture(GL_TEXTURE0);
   GL_ERR_CHECK();
   
   // Set the clear color
   glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
   
   // Set the depth clearing value
   glClearDepth(1.0f);
   
   glPointSize(10.0f);
}

/**
 * Window resize callback
 *
 * @param width   the width of the window
 * @param height  the height of the window
 */
void GLFWCALL resize(int width, int height)
{
   // Set the affine transform of (x,y) from normalized device coordinates to
   // window coordinates. In this case, (-1,1) -> (0, width) and (-1,1) -> (0, height)
   glViewport(0, 0, width, height);
}

/**
 * Keypress callback
 */
void GLFWCALL keypress(int key, int state)
{
   if(state == GLFW_PRESS)
   {
      switch(key)
      {
         case GLFW_KEY_ESC:
            _running = false;
            break;
      }
   }
}

/**
 * Window close callback
 */
int GLFWCALL close(void)
{
   _running = false;
   return GL_TRUE;
}

/**
 * Main loop
 * @param time    time elapsed in seconds since the start of the program
 */
int update(double time)
{
   
   // Clear the color and depth buffers
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   
   glBindVertexArray(_vao);
   glBindTexture(GL_TEXTURE_2D, _texWeightId);
   // Draw the triangle.
   glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)_points.size());
   
   glfwSwapBuffers();

   return GL_TRUE;
}

//END OPENGL stuff

void Visualizer::add(gsl_vector_float *sample){
   
   
   if (data_->mask_ != NULL){
      sample = data_->applyMask(sample);
   }
   
   gsl_vector_float_scale(sample, 1);
   
   int i = (count/across)*imageH;
   int j = count%across*imageW;
   
   for (int ii = 0; ii < imageH; ++ii)
      for (int jj = 0; jj < imageW; ++jj)
      {
         float val = gsl_vector_float_get(sample, (jj+ii*imageW));
         gsl_matrix_float_set(viz, i+ii, j+jj, val);
      }

   count = (count+1)%(across*down);
}

void Visualizer::clear(){
   gsl_matrix_float_set_zero(viz);
   count = 0;
}

void Visualizer::plot(){
   std::string filepath = plotpath + name + "viz.plot";
   FILE *file_handle;
   file_handle = fopen(filepath.c_str(), "w");
   
   for (int i = 0; i < viz->size1; ++i){
      for (int j = 0; j < viz->size2; ++j){
         float val = gsl_matrix_float_get(viz, i, j);
         fprintf(file_handle, "%f ", val);
      }
      fprintf(file_handle, "\n");
   }
   
   fflush(file_handle);
   fclose(file_handle);
}

void Visualizer::initViz(){
   
   int width = imageW*across*3;
   int height = imageH*down*3;
   // Initialize GLFW
   _running = true;
   glfwInit();
   
   // Request an OpenGL core profile context, without backwards compatibility
   glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR,  3);
   glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR,  2);
   glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
   glfwOpenWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
   
   // Open a window and create its OpenGL context
   if(!glfwOpenWindow(width, height, 0,0,0,0, 32,0, GLFW_WINDOW ))
   {
      std::cerr << "Failed to open GLFW window" << std::endl;
      glfwTerminate();
   }
   
   glfwSetWindowSizeCallback(resize);
   glfwSetKeyCallback(keypress);
   glfwSetWindowCloseCallback(close);

   init();
}

void Visualizer::updateViz(){
//   viz->data
   
   glBindTexture(GL_TEXTURE_2D, _texWeightId);
   
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, imageW * across, imageH * down, 0, GL_RED, GL_FLOAT, viz->data);
   GL_ERR_CHECK();
   glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
   GL_ERR_CHECK();
   glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
   GL_ERR_CHECK();
   glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
   GL_ERR_CHECK();
   glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
   GL_ERR_CHECK();
   glActiveTexture(GL_TEXTURE0);
   GL_ERR_CHECK();

   update(glfwGetTime());
}