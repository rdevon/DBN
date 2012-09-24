//
//  CSoftmaxLayer.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/7/12.
//
//

#include "Layers.h"
#include "IO.h"

void CSoftmaxLayer::getExpectations(){
   //Apply continuous softmax.
   for (int i = 0; i < nodenum; ++i){
      for (int j = 0; j < batchsize; ++j){
         float prob = csoftmax(gsl_matrix_float_get(activations, i, j));
         gsl_matrix_float_set(expectations, i, j, prob);
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