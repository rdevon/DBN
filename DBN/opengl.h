#pragma once

#if defined(__APPLE_CC__)
  #ifdef OPENGL3
    #include <OpenGL/gl3.h>
  #else
    #include <OpenGL/gl.h>
    // Apple madness that needs some fixing for cross platform compatibility
    #define glGenVertexArrays(_id, _addr) glGenVertexArraysAPPLE(_id, _addr)
    #define glBindVertexArray(_id) glBindVertexArrayAPPLE(_id)
  #endif
#else
  #include <GL/gl.h>
#endif

#if defined(_WIN32) || defined(_WIN64)
  // Visual Studio upconverts const char* to std::string when doing
  // the following:
  //   std::cerr << "blah blah" << std::endl;
  // For this reason, the string header must be included.
  #include <string>
#endif

#ifndef __APPLE__
// Non-Apple platforms need the GL Extension Wrangler
  #include <GL/glew.h>
  #  if !defined(_WIN32) && !defined(_WIN64)
  //  Assuming Unix, which needs glext.h
  #  include <GL/glext.h>
  #  endif
#else
  // Make sure GLFW knows to include gl3.h header under OS X. This
  // requires the GL/glfw.h be patched, otherwise it will include gl.h
  // and the output from this program will be a black screen. For this
  // to work, glfw.h must be patched. See README.txt for details.
  #define GLFW_GL3
#endif

#ifdef __GNUC__
// There is a bug in version 4.4.5 of GCC on Ubuntu which causes GCC to segfault
// when __PRETTY_FUNCTION__ is used within certain templated functions. 
#  if !(__GNUC__ == 4 && __GNUC_MINOR__ == 4 && __GNUC_PATCHLEVEL__ == 5)
#    define GL_FUNCTION_NAME __PRETTY_FUNCTION__
#  else
#    define GL_FUNCTION_NAME "unknown function"
#  endif
#elif _MSC_VER
#define GL_FUNCTION_NAME __FUNCSIG__
#else
#define GL_FUNCTION_NAME "unknown function"
#endif


extern "C"
{
  // This function does nothing. Called when GL_ASSERT fails and is about to 
  // throw an exception. Put your breakpoint here.
  inline void assert_breakpoint() {}
}


#ifdef _DEBUG
#undef GL_ASSERT
#undef GL_ERR_CHECK

#define GL_ASSERT(_exp, _message) \
{ \
  if(!(_exp)) \
  { \
    assert_breakpoint(); \
    std::ostringstream glassert__out; \
    glassert__out << "Error in file " << __FILE__ << ":" << __LINE__ << "\n";  \
    glassert__out << GL_FUNCTION_NAME << ".\n\n";                                     \
    glassert__out << "Failed expression: " << #_exp << ".\n";                           \
    glassert__out << std::boolalpha << _message << "\n";                                \
    throw std::runtime_error(glassert__out.str());                                         \
  } \
}                                                                  

// Throw an exception if there are any OpenGL errors
#define GL_ERR_CHECK() \
{ \
  GLuint errnum; \
  std::ostringstream gl__out; \
  int n = 0; \
  while((errnum = glGetError()) && n < 10) \
  { \
    if(n == 0) \
    { \
      gl__out << "Error in file " << __FILE__ << ":" << __LINE__ << "\n"; \
      gl__out << GL_FUNCTION_NAME << ".\n\n"; \
    } \
    ++n; \
    gl__out << GL::errorString(errnum) << "\n"; \
  } \
  if(n > 0) \
  { \
    assert_breakpoint(); \
    throw std::runtime_error(gl__out.str()); \
  } \
}

#else
#define GL_ERR_CHECK()
#define GL_ASSERT();
#endif
