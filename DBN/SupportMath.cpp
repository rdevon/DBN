//
//  SupportMath.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "SupportMath.h"
#include <cmath>
#include "SupportFunctions.h"

float sigmoid(float x){
   return (float)1/(1+expf(-x));
}

double softplus(float x){
   if (std::isinf(exp(x))) return x;
   return log(1+exp(x));
}