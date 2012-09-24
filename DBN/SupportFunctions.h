//
//  SupportFunctions.h
//  DBN
//
//  Created by Devon Hjelm on 7/20/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_SupportFunctions_h
#define DBN_SupportFunctions_h
#include "Types.h"

gsl_vector_int *makeShuffleList(int length);
void print_gsl(gsl_vector_float *v);
void print_gsl(gsl_vector_int *v);
void print_gsl(gsl_matrix_float *m);
#endif
