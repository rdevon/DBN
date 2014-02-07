#------------------------------------------------------------------------------
# Sets up the platform specific aspects of the build
# 
# Variables set:
# GLEW_FOUND            True if GLEW has been found
# GLEW_LIBRARIES	Libraries that need to be linked into the executable
# GLEW_INCLUDE_DIRS     Paths to the required include directories
#------------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8)


# Set up platform specific search paths

if(APPLE)
  set(LIBRARY_SEARCH_PATH
    /usr/local
    /opt/local
  )


  set(HEADER_SEARCH_PATH
    /usr/local
    /opt/local
    /usr/local/include
    /opt/local/include
  )


else(APPLE)

  if(WIN32) # Also true on windows 64-bit

    set(LIBRARY_SEARCH_PATH
      "C:/Program Files (x86)/Microsoft SDKs/Windows/v8.0A/Lib"
      "C:/Program Files/Microsoft SDKs/Windows/v8.0a/Lib"
      "C:/cygwin/usr/local"
      "C:/cygwin/lib"
    )

    set(HEADER_SEARCH_PATH
      "C:/Program Files (x86)/Microsoft SDKs/Windows/v8.0a/Include"
      "C:/Program Files/Microsoft SDKs/Windows/v8.0a/Include"
      "C:/cygwin/usr/local/include"
      "C:/cygwin/usr/local/include/freetype2"
      "C:/cygwin/usr/include"
      "C:/cygwin/usr/include/freetype2"
    )


  else(WIN32)

    if(UNIX)
      set(LIBRARY_SEARCH_PATH
        /usr/local
        /opt/local
        /usr
        /opt
      )

      set(HEADER_SEARCH_PATH
        /usr/local
        /opt/local
        /usr
        /opt
        /usr/include
        /opt/include
      )

    else(UNIX)
      # Oh noes!
      message(WARNING "Supported platforms: Unix, OS X and Windows. Your platform isn't supported, but you can still try and build.")
      message(WARNING "Make sure and set LIBRARY_SEARCH_PATH and HEADER_SEARCH_PATH")

    endif(UNIX)
  endif(WIN32)
endif(APPLE)

set(GLEW_FOUND 1)

# Find glew header
find_path(GLEW_INCLUDE_DIR GL/glew.h ${HEADER_SEARCH_PATH})

# Find glew library
if(WIN32)
  find_library(GLEW_LIBRARIES glew32 ${LIBRARY_SEARCH_PATH})
else(WIN32)
  find_library(GLEW_LIBRARIES GLEW ${LIBRARY_SEARCH_PATH})
endif(WIN32)
