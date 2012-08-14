//
//  ReLULayer.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/6/12.
//
//

#include "Layers.h"

void ReLULayer::getExpectations(){
   //Apply softplus.  Might want to pass a general functor later
   for (int i = 0; i < nodenum_; ++i){
      for (int j = 0; j < batchsize_; ++j){
         float preact = gsl_matrix_float_get(activations_, i, j);
         float prob = softplus(preact);
         gsl_matrix_float_set(expectations_, i, j, prob);
      }
   }
}

float ReLULayer :: freeEnergy_contibution() {
   return 0;
}

//The input needs to be shaped depending on the type of visible layer.
void ReLULayer::shapeInput(Input_t* input){
   float min, max;
   gsl_matrix_float_minmax(input, &min, &max);
   gsl_matrix_float_add_constant(input, -min);
   gsl_matrix_float_scale(input, (float)1/((float)(max-min)));
}