//
//  SupportMath.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "SupportMath.h"

float sigmoid(float x){
   return (float)1/(1+expf(-x));
}

double softplus(float x){
   return log(1+exp(x));
}

float gaussian(float x){
   return (float)1/(2*PI)*expf(-pow(x, 2)/(float)2);
}

float sampleNormalDist(float mu, float sigma){
   float u = (float)rand()/(float)RAND_MAX;
   float v = (float)rand()/(float)RAND_MAX;
   float x = sqrt(-(float)2*log(u))*cos((float)2*PI*v);
   float sample = sigma * x + mu;
   return sample;
}