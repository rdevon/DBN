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

class DataSet{
public:
   std::string name;
   int height, width, number, masksize;
   Input_t *train, *test, *validation, *extra;
   
   bool applymask;
   bool denorm;
   
   gsl_vector_float *meanImage;
   gsl_vector_float *mask;
   gsl_vector_float *norm;
   
   DataSet(){
      masksize = 0;
      mask = NULL;
      meanImage = NULL;
   }
   
   void loadMNIST();
   void loadfMRI();
   void loadSPM();
   void loadstim();
   void splitValidate(float percentage = .1);
   void removeMeanImage();
   void getMask();
   void removeMask();
   gsl_vector_float *applyMask(gsl_vector_float *v);
};

#endif
