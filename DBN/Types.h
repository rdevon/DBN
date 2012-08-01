//
//  Types.h
//  DBN
//
//  Created by Devon Hjelm on 7/11/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_Types_h
#define DBN_Types_h

#include <vector>
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_rng.h"
#include <string>


extern std::string MNISTpath;
extern std::string plotpath;
extern std::string fMRIpath;
extern gsl_rng * r;




typedef gsl_matrix_float   Input_t;

typedef enum{DOWNFLAG, UPFLAG} Up_flag_t;

typedef enum{} Input_flag_t;

typedef enum{ACTIVATIONS, PROBABILITIES} Activation_flag_t;

#endif
