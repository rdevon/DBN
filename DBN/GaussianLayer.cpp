//
//  GaussianLayer2.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/6/12.
//
//

#include <iostream>
#include "Layers.h"
#include "IO.h"

GaussianLayer::GaussianLayer(int n) : Layer(n) {
   noise = .2;
   biases = gsl_vector_float_calloc(nodenum);
   vec_update2 = gsl_vector_float_calloc(nodenum);
   stat3 = gsl_matrix_float_alloc(nodenum, batchsize);
   stat4 = gsl_matrix_float_alloc(nodenum, batchsize);
   setsigma = 1;
   for (int i = 0; i < nodenum; ++i)
      gsl_vector_float_set(biases, i, (float)gsl_ran_gaussian(r, 0.01));
   quad_coefficients = gsl_vector_float_alloc(nodenum);
   gsl_vector_float_set_all(quad_coefficients, (float)1/sqrtf(2));
   sigmas = gsl_vector_float_alloc(nodenum);
   gsl_vector_float_set_all(sigmas, 1);
   sigmas = m_factor;
}

void GaussianLayer::makeBatch(int bs){
   Layer::make_batch(bs);
   gsl_matrix_float_free(stat3);
   gsl_matrix_float_free(stat4);
   stat3 = gsl_matrix_float_calloc(nodenum, batchsize);
   stat4 = gsl_matrix_float_calloc(nodenum, batchsize);
}

void GaussianLayer::getExpectations(){
   for (int i = 0; i < nodenum; ++i){
      for (int j = 0; j < batchsize; ++j){
         float expectation = gsl_matrix_float_get(activations, i, j);
         gsl_matrix_float_set(expectations, i, j, expectation);
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
   for (int i = 0; i < nodenum; ++i) {
      float sigma = gsl_vector_float_get(sigmas, i);
      for (int j = 0; j < batchsize; ++j) {
         float exp = (sigma*sigma)*gsl_matrix_float_get(expectations, i, j);
         float sam = (exp + gsl_ran_gaussian(r, sigma));
         gsl_matrix_float_set(samples, i, j, sam);
      }
   }
}

void GaussianLayer::update(ContrastiveDivergence *teacher){
   Layer::update(teacher);
   /*if (0){
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
   }*/
}

void GaussianLayer::shapeInput(DataSet *data){
   Input_t *input = data->train;
   gsl_vector_float *col = gsl_vector_float_alloc(input->size1);
   if (setsigma > 0) {
      data->norm = gsl_vector_float_alloc(data->train->size2);
      data->denorm = true;
      for (int j = 0; j < input->size2; ++j) {
         gsl_matrix_float_get_col(col, input, j);
         float mean = gsl_stats_float_mean(col->data, col->stride, col->size);
         float sd = gsl_stats_float_sd(col->data, col->stride, col->size);
         gsl_vector_float_add_constant(col, -mean);
         gsl_vector_float_scale(col, setsigma/sd);
         gsl_matrix_float_set_col(input, j, col);
         gsl_vector_float_set(data->norm, j, sd);
      }
   }
   else {
      data->denorm = false;
      for (int j = 0; j < input->size2; ++j) {
         float sd = gsl_stats_float_sd(col->data, col->stride, col->size);
         gsl_vector_float_set(sigmas, j, sd);
      }
   }
   gsl_vector_float_free(col);
}

float GaussianLayer::reconstructionCost(gsl_matrix_float *dataMat, gsl_matrix_float *modelMat){
   reconstruction_cost=0;
   gsl_matrix_float *squared_error = gsl_matrix_float_alloc(dataMat->size1, dataMat->size2);
   gsl_matrix_float_memcpy(squared_error, dataMat);
   gsl_matrix_float_sub(squared_error, modelMat);
   gsl_matrix_float_mul_elements(squared_error, squared_error);
   for (int i = 0; i < squared_error->size1; ++i)
      for (int j = 0; j < squared_error->size2; ++j)
         reconstruction_cost+=gsl_matrix_float_get(squared_error, i, j);
   reconstruction_cost /= squared_error->size2;
   gsl_matrix_float_free(squared_error);
   return reconstruction_cost;
}