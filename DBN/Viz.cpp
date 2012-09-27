//
//  Viz.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/24/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include "Viz.h"
#include "Layers.h"
#include "Connections.h"
#include "DBN.h"

//BEGIN OPENGL stuff

bool   _running;             //< true if the program is running, false if it is time to terminate
const char* _vertexSource =
"#version 150\n"
"\n"
"in vec4 vertex;\n"
"in vec2 texCoord;\n"
"uniform mat4 mvp;\n"
"out vec2 tc;\n"
"\n"
"void main(void)\n"
"{\n"
"	gl_Position = mvp * vertex;\n"
"  tc = texCoord;\n"
"}\n";

const char* _fragmentSource =
"#version 150\n"
"\n"
"uniform sampler2D weight;\n"
"uniform vec4 color;\n"
"out vec4 fragColor;\n"
"in vec2 tc;\n"
"#extension GL_ARB_separate_shader_objects : enable\n"
"\n"
"void main(void)\n"
"{\n"
"  fragColor = texture(weight, tc);\n"
"  if (fragColor.r <= -100) fragColor = vec4(1,1,1,1);\n"
"  else if (fragColor.r <= -99) fragColor = vec4(.5,.5,.5,1);\n"
"  else fragColor = fragColor.r * color;\n"
"}\n";

/**
 * Clean up and exit
 *
 * @param exitCode      The exit code, eg, EXIT_SUCCESS or EXIT_FAILURE
 */
void Visualizer::terminate(int exitCode)
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
std::string Visualizer::readTextFile(const std::string& filename)
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
bool Visualizer::shaderCompileStatus(GLuint shader)
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
std::string Visualizer::getShaderLog(GLuint shader)
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
bool Visualizer::programLinkStatus(GLuint program)
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
std::string Visualizer::getProgramLog(GLuint program)
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
GLuint Visualizer::createShader(const std::string& source, GLenum shaderType)
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
GLuint Visualizer::createGLSLProgram()
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
void Visualizer::init(int num_tex_maps, int num_line_plots)
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
   _points.push_back(glm::vec4(-1, -1, 0, 1));
   _points.push_back(glm::vec4(-1,  1, 0, 1));
   _points.push_back(glm::vec4( 1, -1, 0, 1));
   _points.push_back(glm::vec4( 1,  1, 0, 1));
   
   // Texture coordinates for the textured quad
   _texCoords.push_back(glm::vec2(0,1));
   _texCoords.push_back(glm::vec2(0,0));
   _texCoords.push_back(glm::vec2(1,1));
   _texCoords.push_back(glm::vec2(1,0));
   
   // Create the shader program
   _program = createGLSLProgram();
   // Use the shader program that was loaded, compiled and linked
   glUseProgram(_program);
   
   
   // Generate a single handle for a vertex array
   glGenVertexArrays(1, &_vao);
   
   // Bind that vertex array
   glBindVertexArray(_vao);
   
   // Get the location of the "vertex" attribute in the shader program
   _colorLocation = glGetUniformLocation(_program, "color");
   _vertexLocation = glGetAttribLocation(_program, "vertex");
   _texCoordLocation = glGetAttribLocation(_program, "texCoord");
   _weightSamplerLoc = glGetUniformLocation(_program, "weight");
   _mvpLocation = glGetUniformLocation(_program, "mvp");
   
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
                sizeof(glm::vec4) * _points.size(),  // Size in bytes of the buffer
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
                sizeof(glm::vec2) * _texCoords.size(),  // Size in bytes of the buffer
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
   
   _texID.resize(num_tex_maps);
   glGenTextures(num_tex_maps, &_texID[0]);
   
   for(size_t i = 0; i < _texID.size(); ++i)
   {
      glBindTexture(GL_TEXTURE_2D, _texID[i]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _texWidth, _texHeight, 0, GL_RGBA, GL_FLOAT, texels);
      GL_ERR_CHECK();
      glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
      GL_ERR_CHECK();
      glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
      GL_ERR_CHECK();
      glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ); // GL_NEAREST for no interpolation, GL_LINEAR for interpolation
      GL_ERR_CHECK();
      glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
      GL_ERR_CHECK();
      glBindTexture(GL_TEXTURE_2D, 0);
   }
   
   // Generate VAO and buffer objects for line plots
   _lineVAO.resize(num_line_plots);
   _lineBO.resize(num_line_plots);
   glGenVertexArrays(GLsizei(_lineVAO.size()), &_lineVAO[0]);
   glGenBuffers(GLsizei(_lineBO.size()), &_lineBO[0]);
   
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


int Visualizer::update_tex_unit(Unit_Monitor *gl_unit, int id)
{
   glBindTexture(GL_TEXTURE_2D, _texID[id]);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, (int)gl_unit->viz_matrix->size2, (int)gl_unit->viz_matrix->size1, 0, GL_RED, GL_FLOAT, gl_unit->viz_matrix->data);
   //GL_ERR_CHECK();
   
   glm::mat4 mvp;
   
   mvp = glm::translate(glm::mat4(), glm::vec3(gl_unit->x_position, gl_unit->y_position, gl_unit->z_position)) * glm::scale(glm::mat4(), glm::vec3(.001*gl_unit->x_size, .001*gl_unit->y_size, gl_unit->z_size));
   
   glUseProgram(_program);
   glBindVertexArray(_vao);
   
   glActiveTexture(GL_TEXTURE0);
   glUniformMatrix4fv(_mvpLocation, 1, GL_FALSE, glm::value_ptr(mvp));
   glUniform4fv(_colorLocation, 1, gl_unit->color);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)_points.size());
   return 1;
}

int Visualizer::update_line_units(Unit_Monitor *gl_unit, int id){
   
   // Draw some line plots
   std::vector<glm::vec4> _lineData(4);
   
   int x = 0;
   for (auto y:gl_unit->plot_vector){
      _lineData.push_back(glm::vec4(.005*x,2*y,0,1));
      ++x;
   }
   
   glUseProgram(_program);
   glBindVertexArray(_lineVAO[id]);
   glBindBuffer(GL_ARRAY_BUFFER, _lineBO[id]);
   // copy data to gpu
   glBufferData(GL_ARRAY_BUFFER, _lineData.size() * sizeof(glm::vec4), &_lineData[0], GL_STATIC_DRAW);
   // Tell OpenGL where the beginning of the buffer starts for this attribute
   glVertexAttribPointer(_vertexLocation, 4, GL_FLOAT, GL_FLOAT, 0, NULL);
   
   glm::mat4 mvp = glm::translate(glm::mat4(), glm::vec3(gl_unit->x_position, gl_unit->y_position, gl_unit->z_position)) * glm::scale(glm::mat4(), glm::vec3(.05, 0.2/(gl_unit->plot_vector[0]), 1));
   glUniformMatrix4fv(_mvpLocation, 1, GL_FALSE, glm::value_ptr(mvp));
   
   glUniform4fv(_colorLocation, 1, gl_unit->color);
   // Enable the generic vertex attribute array
   glEnableVertexAttribArray(_vertexLocation);
   glDrawArrays(GL_LINE_STRIP, 0, GLsizei(_lineData.size()));
   return 1;
}

//END OPENGL stuff

int Visualizer::update(Learning_Monitor *learning_monitor){
   glfwGetTime();
   
   glClearColor(1, 1, 1, 1);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   
   for (std::map<int, Unit_Monitor*>::iterator gl_iter = learning_monitor->tex_units.begin(); gl_iter != learning_monitor->tex_units.end(); ++gl_iter){
      int id = (*gl_iter).first;
      Unit_Monitor *gl_unit = (*gl_iter).second;
      gl_unit->load_viz();
      gl_unit->scale_matrix_and_threshold();
      update_tex_unit(gl_unit, id);
   }
   
   for (std::map<int, Unit_Monitor*>::iterator gl_iter = learning_monitor->line_units.begin(); gl_iter != learning_monitor->line_units.end(); ++gl_iter){
      int id = (*gl_iter).first;
      Unit_Monitor *gl_unit = (*gl_iter).second;
      gl_unit->load_viz();
      update_line_units(gl_unit, id);
   }
   
   glfwSwapBuffers();
   
   glUseProgram(0);
   
   return GL_TRUE;
}

void Visualizer::open_window(int width, int height){
   
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
   
}

void Visualizer::close_window(){
   GLFWCALL close();
}

void Learning_Monitor::update() {
   
   viz->update(this);
}

// FOR UNIT MONITORS ------------------------------------------------------------------

void Unit_Monitor::add_viz_vector(){
   
   int i = (piece_count/pieces_across)*unit->image_pixels_y;
   int j = piece_count%pieces_across*unit->image_pixels_x;
   
   float max = gsl_stats_float_max(viz_vector->data, 1, viz_vector->size);
   
   for (int ii = 0; ii < unit->image_pixels_y; ++ii)
      for (int jj = 0; jj < unit->image_pixels_x; ++jj)
      {
         float val = gsl_vector_float_get(viz_vector, (jj+ii*unit->image_pixels_x));
         if (val != WHITE && max != 0) val /= max;
         gsl_matrix_float_set(viz_matrix, i+ii, j+jj, val);
      }
   piece_count = (piece_count+1)%(pieces_across*pieces_down);
}

void Unit_Monitor::clear_viz(){
   gsl_matrix_float_set_zero(viz_matrix);
   gsl_vector_float_set_zero(viz_vector);
   piece_count = 0;
}

void Unit_Monitor::finish_setup(){
   viz_vector = gsl_vector_float_alloc((unit->image_pixels_x)*(unit->image_pixels_y));
   viz_matrix = gsl_matrix_float_calloc(unit->image_pixels_y*pieces_down, unit->image_pixels_x*pieces_across);
   z_size = 0;
   color[3] = 1;
}

void Unit_Monitor::plot(){
   std::string filepath = plotpath + name + "viz.plot";
   FILE *file_handle;
   file_handle = fopen(filepath.c_str(), "w");
   
   for (int i = 0; i < viz_matrix->size1; ++i){
      for (int j = 0; j < viz_matrix->size2; ++j){
         float val = gsl_matrix_float_get(viz_matrix, i, j);
         fprintf(file_handle, "%f ", val);
      }
      fprintf(file_handle, "\n");
   }
   fflush(file_handle);
   fclose(file_handle);
}

void Unit_Monitor::load_viz(){}

void Unit_Monitor::scale_matrix_and_threshold(){
   float max = gsl_stats_float_max(viz_matrix->data, 1, viz_matrix->size1*viz_matrix->size2);
   
   for (int i = 0; i < viz_matrix->size1; ++i)
      for (int j = 0; j < viz_matrix->size2; ++j) {
         float val = gsl_matrix_float_get(viz_matrix, i, j);
         
         if (val == WHITE) {}
         else if (max != 0 && val/max >= threshold) gsl_matrix_float_set(viz_matrix, i, j, val/max);
         else gsl_matrix_float_set(viz_matrix, i, j, GREY);
      }
}

// ----- Specific unit monitors


Connection_Learning_Monitor::Connection_Learning_Monitor(Connection* connection){
   Layer *bot = (Layer*)connection->from;
   Layer *top = (Layer*)connection->to;
   bot_sample_monitor = new Layer_Sample_Monitor(bot,500,250,.5);
   top_sample_monitor = new Layer_Sample_Monitor(top,300,250,.5);
   con_weight_monitor = new Connection_Weight_Monitor(connection, 36, 800,400,.5);
   
   top_bias_monitor   = new Layer_Bias_Monitor(top,300);
   
   rec_cost_monitor   = new Reconstruction_Cost_Monitor(bot);
   
   bot_sample_monitor->set_coords(-.55, -.6, 0);
   con_weight_monitor->set_coords(-.1, .1, 0);
   top_sample_monitor->set_coords(-.6, .6, 0);
   
   top_bias_monitor->set_coords(-.6, .8, 0);
   
   rec_cost_monitor->set_coords(-.1, -.8, 0);
   
   tex_units[TOP_SAMPLES] = top_sample_monitor;
   tex_units[BOT_SAMPLES] = bot_sample_monitor;
   tex_units[CONNECTION_WEIGHTS] = con_weight_monitor;
   tex_units[TOP_BIASES] = top_bias_monitor;
   line_units[REC_COST_PLOT] = rec_cost_monitor;
   viz->init(NUM_TEX_MAPS, NUM_LINE_PLOTS);
}

Layer_Bias_Monitor::Layer_Bias_Monitor (Layer* layer, int x_pixels){
   unit = layer;
   name = "layer biases";
   
   threshold = 0;
   z_size = 0;
   color[0] = 1, color[1] = 0, color[2] = 1;
   
   unit->image_pixels_x = layer->nodenum;
   unit->image_pixels_y = 1;
   x_size = x_pixels;
   y_size = 100;
   
   pieces_across = 1;
   pieces_down = y_size;
   finish_setup();
}


/*Feature_Monitor::Feature_Monitor(Layer* feature_layer, Layer* input_layer, int feature){
   
}
*/
void Layer_Bias_Monitor::load_viz(){
   clear_viz();
   
   Layer *layer = (Layer*)unit;
   
   float min_bias, max_bias;
   gsl_vector_float_minmax(layer->biases, &min_bias, &max_bias);
   
   gsl_matrix_float_set_zero(viz_matrix);
   float max_abs_bias = fmaxf(max_bias, -min_bias);
   
   for (int i = 0; i < pieces_down; ++i) {
      float bias_position = (float(i-pieces_down/2)/float(pieces_down/2))*max_abs_bias;
      for (int j = 0; j < unit->image_pixels_x; ++j) {
         float bias = gsl_vector_float_get(layer->biases, j);
         if (bias >= bias_position && bias_position >=0) gsl_matrix_float_set(viz_matrix, i, j, 1);
         else if (bias <= bias_position && bias_position <= 0) gsl_matrix_float_set(viz_matrix, i, j, 1);
      }
   }
}

Layer_Sample_Monitor::Layer_Sample_Monitor (Layer* layer, int x_pixels, int y_pixels, float thresh) {
   unit = layer;
   name = "layer samples";
   threshold = thresh;
   z_size = 0;
   color[0] = 1, color[1] = 1, color[2] = 0;
   pieces_across = pieces_down = 1;
   
   if (layer->input_edge != NULL) {
      unit->image_pixels_x = layer->input_edge->input_x;
      unit->image_pixels_y = layer->input_edge->input_y;
      y_size = y_pixels;
      x_size = (y_size*unit->image_pixels_x)/unit->image_pixels_y;
   }
   else {
      unit->image_pixels_x = layer->nodenum;
      unit->image_pixels_y = 1;
      y_size = 10;
      x_size = x_pixels;
   }
   finish_setup();
}
void Layer_Sample_Monitor::load_viz(){
   clear_viz();
   Layer *layer = (Layer*)unit;
   gsl_matrix_float_get_col(layer->sample_vector, layer->expectations, 0);
   if (layer->input_edge != NULL) {
      layer->input_edge->apply_mask(layer->sample_vector, viz_vector);
   }
   else {
      for (int i = 0; i < layer->sample_vector->size; ++i) {
         float val = gsl_vector_float_get(layer->sample_vector, i);
         gsl_vector_float_set(viz_vector, i, val);
      }
   }
   
   add_viz_vector();
}

Connection_Weight_Monitor::Connection_Weight_Monitor (Connection* connection, int sample_num, int x_pixels, int y_pixels, float thresh) :sample_number(sample_num) {
   unit = connection;
   name = "connection weights";
   threshold = thresh;
   z_size = 0;
   color[0] = 0, color[1] = 1, color[2] = 1;
   
   if (connection->from->input_edge != NULL) {
      unit->image_pixels_x = connection->from->input_edge->input_x;
      unit->image_pixels_y = connection->from->input_edge->input_y;
   }
   else {
      unit->image_pixels_x = ((Layer*)connection->from)->nodenum;
      unit->image_pixels_y = ((Layer*)connection->to)->nodenum;
   }
   
   y_size = y_pixels;
   x_size = x_pixels;
   
   pieces_across = 9;
   pieces_down = 4;
   
   finish_setup();
}

void Connection_Weight_Monitor::load_viz(){
   Connection *connection = (Connection*)unit;
   clear_viz();
   
   gsl_vector_float *sums = gsl_vector_float_alloc(connection->weights->size1);
   for (int i = 0; i < connection->weights->size1; ++i){
      float sum = 0;
      for (int j = 0; j < connection->weights->size2; ++j)
         sum += pow(gsl_matrix_float_get(connection->weights, i , j),2);
         //sum += gsl_matrix_float_get(connection->weights, i, j);
      Layer *top = ((Layer*)connection->to);
      float bias = gsl_vector_float_get(top->biases, i);
      gsl_vector_float_set(sums, i, sum*bias);
   }
   
   gsl_permutation *p = gsl_permutation_alloc(connection->weights->size1);
   gsl_sort_vector_float_index(p, sums);
   
   for (int i = 0; i < sample_number; ++i) {
      //std::cout << connection->weights->size1 - i << " " << i << std::endl;
      size_t index = p->data[connection->weights->size1 - i - 1];
      gsl_matrix_float_get_row(connection->node_projections, connection->weights, index);
      if (connection->from->input_edge != NULL) connection->from->input_edge->apply_mask(connection->node_projections, viz_vector);
      add_viz_vector();
   }
   gsl_vector_float_free(sums);
   gsl_permutation_free(p);
}

/*
Pathway_Monitor::Pathway_Monitor (Pathway* pathway, int across, int down, float sc, float v_scale, float thresh)
: Unit_Monitor(pieces_across, pieces_down, threshold) {
   unit = pathway;
   unit->monitor = this;
   name = "pathway samples";
   
   Layer *base = (Layer*)pathway->path[0]->from;
   
   if (base->input_edge != NULL) {
      unit->image_pixels_x = base->input_edge->input_x;
      unit->image_pixels_y = base->input_edge->input_y;
   }
   else {
      unit->image_pixels_x = sqrt(base->nodenum);
      unit->image_pixels_y = (base->nodenum + unit->image_pixels_x + 1)/unit->image_pixels_x;
   }
   
   finish_setup();
}

void Pathway_Monitor::load_viz(){
   clear_viz();
   Pathway *pathway = (Pathway*)unit;
   pathway->set_status_all(WAITING);
   pathway->first = pathway->path[0];
   if (pathway->last == NULL) pathway->last = pathway->path.back();;
   
   Layer *top = (Layer*)pathway->last->to;
   top->status = DONE;
   Layer *bot = (Layer*)pathway->first->from;
   int old_batch = top->batchsize;
   pathway->make_batch(top->nodenum);
   gsl_matrix_float_set_identity(top->samples);
   pathway->transmit_down();
   
   for (int j = 0; j < top->nodenum; ++j){
      gsl_matrix_float_get_col(viz_vector, bot->samples, j);
      add_viz_vector();
   }
   pathway->make_batch(old_batch);
}
*/


Reconstruction_Cost_Monitor::Reconstruction_Cost_Monitor(Layer *layer){
   unit = layer;
   name = "layer reconstruction cost";
}

void Reconstruction_Cost_Monitor::load_viz(){
   Layer *layer = (Layer*)unit;
   plot_vector.push_back(layer->reconstruction_cost);
   color[0] = 1, color[1] = 0, color[2] = 0, color[3] = 1;
   x_size = y_size = z_size = 1;
}

