//
//  SupportMath.h
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_SupportMath_h
#define DBN_SupportMath_h
#include <math.h>
#include "Types.h"

float sigmoid(float x);
double softplus(float x);
float gaussian(float x);
float softmax(float x, int classes);

float sampleNormalDist(float mu, float sigma);
#endif
