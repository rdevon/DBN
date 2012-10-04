#------------------------------------------------------------------------------
# Sets up the platform specific aspects of the build
# # Variables set:
#
# GLM_INCLUDE_DIRS     Paths to the required include directories
#------------------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)

# Set up platform specific search paths
if(APPLE)
  set(HEADER_SEARCH_PATH
    /usr/local
    /opt/local
    /usr/local/include
    /opt/local/include
  )

else(APPLE)

  if(WIN32) # Also true on windows 64-bit

    set(HEADER_SEARCH_PATH
      "C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0a/Include"
      "C:/Program Files/Microsoft SDKs/Windows/v7.0a/Include"
      "C:/cygwin/usr/local/include"
      "C:/cygwin/usr/local/include/freetype2"
      "C:/cygwin/usr/include"
      "C:/cygwin/usr/include/freetype2"
    )

  else(WIN32)

    if(UNIX)

      set(HEADER_SEARCH_PATH
        /usr/local/include
        /opt/local/include
        /usr/local
        /opt/local
        /usr
        /opt
      )

    else(UNIX)

      # Oh noes!
      message(WARNING "Supported platforms: Unix, OS X and Windows. Your platform isn't supported, but you can still try and build.")
      message(WARNING "Make sure and set LIBRARY_SEARCH_PATH and HEADER_SEARCH_PATH")
    endif(UNIX)
  endif(WIN32)
endif(APPLE)

# Find glfw header
find_path(GLM_INCLUDE_DIR glm/glm.hpp ${HEADER_SEARCH_PATH})
