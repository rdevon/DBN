//
//  GaussianLayer2.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/6/12.
//
//

#include <iostream>
#include "Layers.h"

GaussianLayer::GaussianLayer(int n) : Layer(n) {
   setsigma = 1;
   biases_ = gsl_vector_float_calloc(nodenum_);
   for (int i = 0; i < nodenum_; ++i)
      gsl_vector_float_set(biases_, i, (float)gsl_ran_gaussian(r, 0.01));
   quad_coefficients = gsl_vector_float_alloc(nodenum_);
   gsl_vector_float_set_all(quad_coefficients, (float)1/sqrtf(2));
   sigmas = gsl_vector_float_alloc(nodenum_);
   gsl_vector_float_set_all(sigmas, 1);
   vec_update2 = gsl_vector_float_calloc(nodenum_);
   stat3 = gsl_matrix_float_alloc(nodenum_, batchsize_);
   stat4 = gsl_matrix_float_alloc(nodenum_, batchsize_);
}

void GaussianLayer::makeBatch(int batchsize){
   Layer::makeBatch(batchsize);
   stat3 = gsl_matrix_float_calloc(nodenum_, batchsize_);
   stat4 = gsl_matrix_float_calloc(nodenum_, batchsize_);
}

void GaussianLayer::getExpectations(){
   for (int i = 0; i < nodenum_; ++i){
      float sigma = gsl_vector_float_get(sigmas, i);
      float quad = gsl_vector_float_get(quad_coefficients, i);
      for (int j = 0; j < batchsize_; ++j){
         float expectation = gsl_matrix_float_get(activations_, i, j)/(sigma*sigma);
         gsl_matrix_float_set(expectations_, i, j, expectation);
      }
   }
}

void GaussianLayer::getSigmas(){
   for (int i = 0; i < quad_coefficients->size; ++i){
      float q = gsl_vector_float_get(quad_coefficients, i);
      gsl_vector_float_set(sigmas, i, (float)1/(sqrtf(2)*q));
   }
}

void GaussianLayer::sample(){
   for (int i = 0; i < nodenum_; ++i) {
      float sigma = gsl_vector_float_get(sigmas, i);
      for (int j = 0; j < batchsize_; ++j) {
         float exp = (sigma*sigma)*gsl_matrix_float_get(expectations_, i, j);
         float sam = exp + gsl_ran_gaussian(r, sigma);
         //float sam = gsl_ran_gaussian(exp, sigma);
         gsl_matrix_float_set(samples_, i, j, sam);
      }
   }
}

void GaussianLayer::update(ContrastiveDivergence *teacher){
   Layer::update(teacher);
   if (setsigma <= 0){
      float learning_rate = teacher->learningRate_/(float)(teacher->batchsize_*teacher->batchsize_);
      
      gsl_vector_float *quad_update_pos = gsl_vector_float_alloc(nodenum_);
      gsl_vector_float *quad_update_neg = gsl_vector_float_alloc(nodenum_);
      
      gsl_matrix_float_memcpy(stat3, stat1);
      gsl_matrix_float_memcpy(stat4, stat2);
      
      gsl_blas_sgemv(CblasNoTrans, 1, stat3, teacher->identity, 0, quad_update_pos);
      gsl_vector_float_mul(quad_update_pos, quad_update_pos);
      
      gsl_blas_sgemv(CblasNoTrans, 1, stat4, teacher->identity, 0, quad_update_neg);
      gsl_vector_float_mul(quad_update_neg, quad_update_neg);
      
      gsl_vector_float_sub(quad_update_neg, quad_update_pos);
      
      gsl_vector_float_mul(quad_update_neg, quad_coefficients);
      gsl_vector_float_scale(quad_update_neg, 2*learning_rate);
      gsl_vector_float_add(quad_coefficients, quad_update_neg);
      
      for (int i = 0; i < quad_coefficients->size; ++i){
         if (gsl_vector_float_get(quad_coefficients, i) < .1) gsl_vector_float_set(quad_coefficients, i, .01);
      }
      
      getSigmas();
      
      gsl_vector_float_memcpy(vec_update2, quad_coefficients);
      
      gsl_vector_float_free(quad_update_neg);
      gsl_vector_float_free(quad_update_pos);
   }
}

void GaussianLayer::shapeInput(Input_t* input){
   if (setsigma > 0) {
      gsl_vector_float *col = gsl_vector_float_alloc(input->size1);
      for (int j = 0; j < input->size2; ++j) {
         gsl_matrix_float_get_col(col, input, j);
         float mean = gsl_stats_float_mean(col->data, col->stride, col->size);
         float sd = gsl_stats_float_sd(col->data, col->stride, col->size);
         gsl_vector_float_add_constant(col, -mean);
         gsl_vector_float_scale(col, setsigma/sd);
         gsl_matrix_float_set_col(input, j, col);
      }
      gsl_vector_float_free(col);
   }
}

