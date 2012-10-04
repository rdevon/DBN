cmake_minimum_required(VERSION 2.8)

set(PROJ_NAME DBN)

project(${PROJ_NAME})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

include(FindOpenGL)
include(FindGLFW)
include(FindGLM)
include(FindGLEW)
include(FindGSL)

# Make sure that OpenGL is found
if(NOT OPENGL_FOUND)
  message(ERROR "Could not find OpenGL")
endif(NOT OPENGL_FOUND)

# Make sure that GLFW is found
if(NOT GLFW_FOUND)
  message(ERROR "Could not find GLFW")
endif(NOT GLFW_FOUND)

if(NOT GSL_FOUND)
  message(ERROR "Could not find GSL")
endif(NOT GSL_FOUND)

# Use OpenGL 3 core context
add_definitions("-DGLFW_INCLUDE_GL3 -DGLFW_NO_GLU -DOPENGL3")

# Set the include directories
include_directories(
  ${OPENGL_INCLUDE_DIR}
  ${GLFW_INCLUDE_DIR}
  ${GLM_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${GSL_INCLUDE_DIRS}
  ${GSLCBLAS_INCLUDE_DIRS}
)

# Get the path to the source code and create a define. This is used
# for locating the shaders
add_definitions("-DSOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
add_definitions("-std=c++0x")

# Platform specific libraries
if(APPLE)
  set(PLATFORM_LIBRARIES "-framework IOKit")
else(APPLE)
  set(PLATFORM_LIBRARIES ${GLEW_LIBRARIES})
endif(APPLE)

set(SOURCE_FILES
   DBN.cpp
   GaussianLayer.cpp
   GradientDescent.cpp
   IO.cpp
   Layers.cpp
   Connections.cpp
   main.cpp
   MLP.cpp
   Pathway.cpp
   RBM.cpp
   ReLULayer.cpp
   SigmoidLayer.cpp
   SoftmaxLayer.cpp
   SupportFunctions.cpp
   SupportMath.cpp
   Teacher.cpp
   Timecourses.cpp
   Types.cpp
   Viz.cpp
)

set(HEADER_FILES
   Connections.h
   DBN.h
   GradientDescent.h
   IO.h
   Layers.h
   Connections.h
   MLP.h
   opengl.h
   Pathway.h
   RBM.h
   SupportFunctions.h
   SupportMath.h
   Teacher.h
   Timecourses.h
   Types.h
   Viz.h
)

# Add a target executable
add_executable(${PROJ_NAME}
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

# Libraries to be linked
target_link_libraries(${PROJ_NAME}
  ${OPENGL_LIBRARIES}
  ${GLFW_LIBRARIES}
  ${PLATFORM_LIBRARIES}
  ${GSL_LIBRARIES}
  ${GSLCBLAS_LIBRARIES}
)
