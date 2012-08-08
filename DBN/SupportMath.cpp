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

double softplus(double x){
   return log(1+exp(x));
}

float gaussian(float x){
   return (float)1/(2*PI)*expf(-pow(x, 2)/(float)2);
}

float csoftmax(double x){
   return expf(x)/(expf(1)-1);
}