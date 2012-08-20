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
         float act = gsl_matrix_float_get(activations_, i, j);
         float prob;
         //if (act < 100)
            prob = softplus(act);
         //else prob = act;
         gsl_matrix_float_set(expectations_, i, j, prob);
      }
   }
}

void ReLULayer::sample(){
   // Sample = max(0, x+N(0,sigmoid(x)))
   for (int i = 0; i < nodenum_; ++i){
      for (int j = 0; j < batchsize_; ++j){
         float exp = gsl_matrix_float_get(expectations_, i, j);
         float sam = fmaxf(0, exp + gsl_ran_gaussian(r, sigmoid(exp)));
         gsl_matrix_float_set(samples_, i, j, sam);
      }
   }
   //print_gsl(expectations_);
   
}

void ReLULayer::update(ContrastiveDivergence *teacher){
   Layer::update(teacher);
}

float ReLULayer :: freeEnergy_contibution() {
   return 0;
}

//The input needs to be shaped depending on the type of visible layer.
void ReLULayer::shapeInput(DataSet* data){
   Input_t *input = data->train;
   float min, max;
   gsl_matrix_float_minmax(input, &min, &max);
   gsl_matrix_float_add_constant(input, -min);
   gsl_matrix_float_scale(input, (float)1/((float)(max-min)));
}