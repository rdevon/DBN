//
//  CSoftmaxLayer.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/7/12.
//
//

#include "Layers.h"

void CSoftmaxLayer::getExpectations(){
   //Apply continuous softmax.
   for (int i = 0; i < nodenum_; ++i){
      for (int j = 0; j < batchsize_; ++j){
         float prob = csoftmax(gsl_matrix_float_get(activations_, i, j));
         gsl_matrix_float_set(expectations_, i, j, prob);
      }
   }
}

float CSoftmaxLayer :: freeEnergy_contibution() {return 0;}

void CSoftmaxLayer::sample(){}

//The input needs to be shaped depending on the type of visible layer.
void CSoftmaxLayer::shapeInput(DataSet *data){
   Input_t *input = data->train;
   float min, max;
   gsl_matrix_float_minmax(input, &min, &max);
   gsl_matrix_float_add_constant(input, -min);
   gsl_matrix_float_scale(input, (float)1/((float)(max-min)));
}