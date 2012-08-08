//
//  GaussianLayer2.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/6/12.
//
//

#include <iostream>
#include "Layers.h"

void GaussianLayer::setProbs(){
   for (int i = 0; i < nodenum_; ++i)
      for (int j = 0; j < batchsize_; ++j){
         float preact = gsl_matrix_float_get(preactivations_, i, j);
         float probability = gaussian(preact);
         gsl_matrix_float_set(probabilities_, i, j, probability);
      }
}

void GaussianLayer::shapeInput(Input_t* input){
   
   gsl_vector_float *col = gsl_vector_float_alloc(input->size1);
   for (int j = 0; j < input->size2; ++j) {
      gsl_matrix_float_get_col(col, input, j);
      float mean = gsl_stats_float_mean(col->data, col->stride, col->size);
      float variance = gsl_stats_float_sd(col->data, col->stride, col->size);
      gsl_vector_float_add_constant(col, -mean);
      gsl_vector_float_scale(col, (float)1/variance);
      gsl_matrix_float_set_col(input, j, col);
      mean = gsl_stats_float_mean(col->data, col->stride, col->size);
      variance = gsl_stats_float_variance(col->data, col->stride, col->size);
   }
   gsl_vector_float_free(col);
}