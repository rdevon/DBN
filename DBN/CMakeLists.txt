cmake_minimum_required(VERSION 2.8)

set(PROJ_NAME DBN)

project(${PROJ_NAME})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_definitions("-DSOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
add_definitions("-DBIN_DIR=\"${CMAKE_CURRENT_BINARY_DIR}\"")
add_definitions("-DOUT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
add_definitions("-DOPENGL3")
add_definitions("-std=c++0x")

SET(USE_OPENGL 0 CACHE BOOL "Use Visualization?")

include(FindGSL)

if(NOT GSL_FOUND)
  message(ERROR "Could not find GSL")
endif(NOT GSL_FOUND)

if(USE_OPENGL)
include(FindOpenGL)
include(FindGLFW)
include(FindGLM)
include(FindGLEW)
include(FindBLAS)

# Make sure that OpenGL is found
if(NOT OPENGL_FOUND)
  message(ERROR "Could not find OpenGL")
endif(NOT OPENGL_FOUND)

# Make sure that GLFW is found
if(NOT GLFW_FOUND)
  message(ERROR "Could not find GLFW")
endif(NOT GLFW_FOUND)

# Make sure that blas is found
if(NOT BLASL_FOUND)
  message(ERROR "Could not find Blas")
endif(NOT BLAS_FOUND)


# Use OpenGL 3 core context
add_definitions("-DGLFW_INCLUDE_GL3 -DGLFW_NO_GLU -DOPENGL3")
set(OPENGL_INCLUDE_DIRECTORIES
  ${OPENGL_INCLUDE_DIR}
  ${GLFW_INCLUDE_DIR}
  ${GLM_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
)

set(OPENGL_LIBS
  ${OPENGL_LIBRARIES}
  ${GLFW_LIBRARIES}
)
add_definitions("-DUSEGL")
endif(USE_OPENGL)

# Had to add hdf5 linking directly
if(APPLE)
  set(HDF5_INCLUDE_DIRS
    /usr/local/include
  )

  set(HDF5_LIBRARIES
    "-lhdf5"
  )

  set(HDF5_CPP_LIBRARIES
    "-lhdf5_cpp"
  )

else(APPLE)
  set(HDF5_INCLUDE_DIRS
    /usr/include
  )
  set(HDF5_LIBRARIES
    "-lhdf5"
  )
  set(HDF5_CPP_LIBRARIES
    "-lhdf5_cpp"
  )
  set(ATLAS_LIBRARIES
    "-lm -L/na/homes/dhjelm/My_Libs/ -lcblas -latlas"
  )
endif(APPLE)  


# Set the include directories
include_directories(
  ${OPENGL_INCLUDE_DIRECTORIES}
  ${GSL_INCLUDE_DIRS}
  ${CBLAS_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIRS}
)

# Get the path to the source code and create a define. This is used
# for locating the shaders
add_definitions("-DSOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
add_definitions("-DOPENGL3")
add_definitions("-std=c++0x")

# Platform specific libraries
if(APPLE)
  set(PLATFORM_LIBRARIES "-framework IOKit")
else(APPLE)
  set(PLATFORM_LIBRARIES ${GLEW_LIBRARIES})
endif(APPLE)

set(SOURCE_FILES
   DBN.cpp
   Autoencoder.cpp
   GaussianLayer.cpp
   IO.cpp
   DataSets.cpp
   MNIST.cpp
   fMRI_Slice.cpp
   fMRI_full.cpp
   SPM.cpp
   Layers.cpp
   Connections.cpp
   main.cpp
   MLP.cpp
   RBM.cpp
   ReLULayer.cpp
   SigmoidLayer.cpp
   SoftmaxLayer.cpp
   Timecourses.cpp
   SupportFunctions.cpp
   SupportMath.cpp
   Teacher.cpp
   Params.cpp
   Viz.cpp
   Viz_Units.cpp
   Monitors.cpp
   Monitor_Units.cpp
   MNIST.cpp
   Get_Network.cpp
   Matrix.cpp
   Tests.cpp
)

set(HEADER_FILES
   Autoencoder.h
   Connections.h
   DBN.h
   IO.h
   DataSets.h
   Layers.h
   Connections.h
   MLP.h
   opengl.h
   RBM.h
   SupportFunctions.h
   SupportMath.h
   Teacher.h
   Timecourses.h
   Params.h
   Viz.h
   Viz_Units.h
   Monitors.h
   Monitor_Units.h
   Get_Network.h
   Matrix.h
   Tests.h
)

# Add a target executable
add_executable(${PROJ_NAME}
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

# Libraries to be linked
target_link_libraries(${PROJ_NAME}
  ${OPENGL_LIBS}
  ${PLATFORM_LIBRARIES}
  ${GSL_LIBRARIES}
  ${CBLAS_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${HDF5_CPP_LIBRARIES}
  ${ATLAS_LIBRARIES}
)
