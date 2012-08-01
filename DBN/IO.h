//
//  IO.h
//  DBN
//
//  Created by Devon Hjelm on 7/23/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_IO_h
#define DBN_IO_h
#include "Types.h"
#include <fstream>
#include <sys/stat.h>
#include <fstream>
#include "dirent.h"
#include <sstream>
#include <gsl/gsl_statistics.h>

class DataSet{
public:
   std::string name;
   int height, width, number;
   Input_t *train, *test, *validation;
   
   gsl_vector_float *meanImage_;
   gsl_vector_float *mask_;
   
   DataSet(){}
   
   void loadMNIST();
   void loadfMRI();
   
   void splitValidate(float percentage = .1);
   void removeMeanImage();
   void getMask();
   void removeMask();
   void addMask();
};

#endif
