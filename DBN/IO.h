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
   int height, width, number, masksize_;
   Input_t *train, *test, *validation, *extra;
   
   bool applymask;
   bool denorm;
   
   gsl_vector_float *meanImage_;
   gsl_vector_float *mask_;
   gsl_vector_float *norm_;
   
   DataSet(){
      masksize_ = 0;
      mask_ = NULL;
      meanImage_ = NULL;
   }
   
   void loadMNIST();
   void loadfMRI();
   void loadSPM();
   void splitValidate(float percentage = .1);
   void removeMeanImage();
   void getMask();
   void removeMask();
   gsl_vector_float *applyMask(gsl_vector_float *v);
};

#endif
