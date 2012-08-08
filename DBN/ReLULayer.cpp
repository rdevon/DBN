//
//  ReLULayer.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/6/12.
//
//

#include "Layers.h"

void ReLULayer::setProbs(){
   //Apply softplus.  Might want to pass a general functor later
   for (int i = 0; i < nodenum_; ++i){
      for (int j = 0; j < batchsize_; ++j){
         float preact = gsl_matrix_float_get(preactivations_, i, j);
         float prob = softplus(preact);
         gsl_matrix_float_set(probabilities_, i, j, prob);
      }
   }
}

void ReLULayer :: getFreeEnergy() {
}

//The input needs to be shaped depending on the type of visible layer.
void ReLULayer::shapeInput(Input_t* input){
   float min, max;
   gsl_matrix_float_minmax(input, &min, &max);
   gsl_matrix_float_add_constant(input, -min);
   gsl_matrix_float_scale(input, (float)1/((float)(max-min)));
}