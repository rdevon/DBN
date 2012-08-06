//
//  GaussianLayer2.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/6/12.
//
//

#include <iostream>
#include "Layers.h"

void Activator::activateGaussianLayer2(GaussianLayer2* s, Layer* layer, Up_flag_t up){
   gsl_matrix_float *weights;
   CBLAS_TRANSPOSE_t transFlag;
   
   //Check flag for up or down activation
   if (up) {
      weights = layer->weights_;
      transFlag = CblasNoTrans;
   }
   
   else {
      weights = s->weights_;
      transFlag = CblasTrans;
   }
   
   //Expand the sigmoid biases into a matrix for matrix operation
   for (int j = 0; j < s->batchsize_; ++j) gsl_matrix_float_set_col(s->batchbiases_, j, s->biases_);
   
   //Compute x = W^T v + b or W v + b depending on up or down activation
   gsl_blas_sgemm(transFlag, CblasNoTrans, 1, weights, layer->activations_, 0, s->preactivations_);
   gsl_matrix_float_add(s->preactivations_, s->batchbiases_);
   
   for (int i = 0; i < s->nodenum_; ++i)
      for (int j = 0; j < s->batchsize_; ++j){
         float preact = gsl_matrix_float_get(s->preactivations_, i, j);
         float mean = gaussian(preact);
         gsl_matrix_float_set(s->activations_, i, j, mean);
      }
   
   //Sample if sample flag on
   if (flag_ == ACTIVATIONS){
      for (int i = 0; i < s->nodenum_; ++i){
         for (int j = 0; j < s->batchsize_; ++j){
            float u = gsl_rng_uniform(r);
            float act = (float)(gsl_matrix_float_get(s->means_, i, j) > u);
            gsl_matrix_float_set(s->activations_, i, j, act);
         }
      }
   }
   else gsl_matrix_float_memcpy(s->activations_, s->means_);
}

void GaussianLayer2::activate(Activator& act, Layer* layer, Up_flag_t UFLAG){
   act.activateGaussianLayer2(this, layer, UFLAG);
}

void GaussianLayer2::shapeInput(Input_t* input){
   
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