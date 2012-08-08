//
//  CSoftmaxLayer.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/7/12.
//
//

#include "Layers.h"

void CSoftmaxLayer::setProbs(){
   //Apply continuous softmax.
   for (int i = 0; i < nodenum_; ++i){
      for (int j = 0; j < batchsize_; ++j){
         float mean = csoftmax(gsl_matrix_float_get(preactivations_, i, j));
         gsl_matrix_float_set(probabilities_, i, j, mean);
      }
   }
}

void CSoftmaxLayer :: getFreeEnergy() {}

//This may seem superflurous, but it will be useful to have visitor function when there are multiple activation sources

//The input needs to be shaped depending on the type of visible layer.
void CSoftmaxLayer::shapeInput(Input_t* input){
   float min, max;
   gsl_matrix_float_minmax(input, &min, &max);
   gsl_matrix_float_add_constant(input, -min);
   gsl_matrix_float_scale(input, (float)1/((float)(max-min)));
}