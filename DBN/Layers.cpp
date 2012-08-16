//
//  Layers.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/18/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "Layers.h"


Layer::Layer(int nodenum) : nodenum_(nodenum), batchsize_(1), frozen(false), energy_(0) {
   
   activations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   expectations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   samples_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   batchbiases_ = gsl_matrix_float_alloc(nodenum_, batchsize_);
   
   vec_update = gsl_vector_float_calloc(nodenum_);
   mat_update = gsl_matrix_float_calloc(nodenum_, batchsize_);
   stat1 = gsl_matrix_float_alloc(nodenum_, batchsize_);
   stat2 = gsl_matrix_float_alloc(nodenum_, batchsize_);
}

void Layer::clear(){
   frozen = false;
   expectation_up_to_date = false;
   sample_up_to_date = false;
   makeBatch(1);
   energy_ = 0;
}

void Layer::makeBatch(int batchsize){
   
   //For batch processing.
   gsl_matrix_float_free(activations_);
   gsl_matrix_float_free(expectations_);
   gsl_matrix_float_free(samples_);
   gsl_matrix_float_free(batchbiases_);
   gsl_matrix_float_free(stat1);
   gsl_matrix_float_free(stat2);
   
   batchsize_ = batchsize;
   
   activations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   expectations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   samples_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   batchbiases_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   stat1 = gsl_matrix_float_calloc(nodenum_, batchsize_);
   stat2 = gsl_matrix_float_calloc(nodenum_, batchsize_);
   
}

void Layer::expandBiases(){
   for (int j = 0; j < batchsize_; ++j) gsl_matrix_float_set_col(batchbiases_, j, biases_);
}

void Layer::update(ContrastiveDivergence *teacher){
   gsl_vector_float *bias_update = vec_update;
   float learning_rate = teacher->learningRate_/(float)teacher->batchsize_;
   gsl_blas_sgemv(CblasNoTrans, learning_rate, stat1, teacher->identity, teacher->momentum_, bias_update);
   gsl_blas_sgemv(CblasNoTrans, -learning_rate, stat2, teacher->identity, 1, bias_update);
   gsl_vector_float_add(biases_, bias_update);
}