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
   biases_ = gsl_vector_float_calloc(nodenum_);
   for (int i = 0; i < nodenum_; ++i)
      gsl_vector_float_set(biases_, i, (float)gsl_ran_gaussian(r, 0.01));
   quad_coefficients = gsl_vector_float_alloc(nodenum_);
   gsl_vector_float_set_all(quad_coefficients, 1);
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
      float quad = gsl_vector_float_get(quad_coefficients, i);
      for (int j = 0; j < batchsize_; ++j){
         float expectation = gsl_matrix_float_get(activations_, i, j)/(2*quad*quad);
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
         float exp = gsl_matrix_float_get(expectations_, i, j);
         float sam = gsl_ran_gaussian_pdf(exp, sigma);
         gsl_matrix_float_set(samples_, i, j, sam);
      }
   }
}

void GaussianLayer::update(ContrastiveDivergence *teacher){
   gsl_vector_float *bias_update = vec_update;
   float learning_rate = teacher->learningRate_/(float)teacher->batchsize_;
   gsl_blas_sgemv(CblasNoTrans, learning_rate, stat1, teacher->identity, teacher->momentum_, bias_update);
   gsl_blas_sgemv(CblasNoTrans, -learning_rate, stat2, teacher->identity, 1, bias_update);
   gsl_vector_float_add(biases_, bias_update);

   learning_rate /= (float)teacher->batchsize_;
   
   gsl_vector_float *quad_update = vec_update2;
   
   gsl_matrix_float_memcpy(stat3, stat1);
   gsl_matrix_float_memcpy(stat4, stat2);
   
   gsl_matrix_float_mul_elements(stat3, stat3);
   gsl_matrix_float_mul_elements(stat4, stat4);
   
   gsl_blas_sgemv(CblasNoTrans, 1, stat4, teacher->identity, 0, quad_update);
   gsl_blas_sgemv(CblasNoTrans, -2*learning_rate, stat3, teacher->identity, 2*learning_rate, quad_update);
   
   gsl_vector_float_mul(quad_update, quad_coefficients);
   gsl_vector_float_add(quad_coefficients, quad_update);
   
   for (int i = 0; i < quad_coefficients->size; ++i){
      if (gsl_vector_float_get(quad_coefficients, i) < .1) gsl_vector_float_set(quad_coefficients, i, .0001);
   }
   
   getSigmas();
   
   gsl_vector_float_memcpy(vec_update2, sigmas);
}

void GaussianLayer::shapeInput(Input_t* input){
   
   gsl_vector_float *col = gsl_vector_float_alloc(input->size1);
   for (int j = 0; j < input->size2; ++j) {
      gsl_matrix_float_get_col(col, input, j);
      float mean = gsl_stats_float_mean(col->data, col->stride, col->size);
      gsl_vector_float_add_constant(col, -mean);
      gsl_matrix_float_set_col(input, j, col);
      //float sd = gsl_stats_float_sd(col->data, col->stride, col->size);
      //gsl_vector_float_set(quad_coefficients, j, (float)1/(sqrtf(2)*sd));
   }
   getSigmas();
   gsl_vector_float_free(col);
}

