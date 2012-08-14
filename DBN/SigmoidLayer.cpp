//
//  SigmoidLayer.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/18/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//


#include "Layers.h"
void SigmoidLayer::getExpectations(){
   //Apply sigmoid.  Might want to pass a general functor later
   for (int i = 0; i < nodenum_; ++i){
      for (int j = 0; j < batchsize_; ++j){
         float exp = sigmoid(gsl_matrix_float_get(activations_, i, j));
         gsl_matrix_float_set(expectations_, i, j, exp);
      }
   }
}

void SigmoidLayer::sample(){
   for (int i = 0; i < nodenum_; ++i){
      for (int j = 0; j < batchsize_; ++j){
         float u = gsl_rng_uniform(r);
         float sample = (float)(gsl_matrix_float_get(expectations_, i, j) > u);
         gsl_matrix_float_set(samples_, i, j, sample);
      }
   }
}

void SigmoidLayer::update(ContrastiveDivergence *teacher){
   gsl_vector_float *bias_update = vec_update;
   float learning_rate = teacher->learningRate_/(float)teacher->batchsize_;
   gsl_blas_sgemv(CblasNoTrans, learning_rate, stat1, teacher->identity, teacher->momentum_, bias_update);
   gsl_blas_sgemv(CblasNoTrans, -learning_rate, stat2, teacher->identity, 1, bias_update);
   gsl_vector_float_add(biases_, bias_update);
}

float SigmoidLayer::reconstructionCost(gsl_matrix_float *dataMat, gsl_matrix_float *modelMat){
   double RE = 0;
   for (int i = 0; i < nodenum_; ++i){
      for (int j = 0; j < batchsize_; ++j){
         double dataAct = gsl_matrix_float_get(dataMat, i, j);
         double modelAct = gsl_matrix_float_get(modelMat, i, j);
         RE += dataAct * log(modelAct) + (1-dataAct)*log(1- modelAct);
         //std::cout << RE << " " << dataAct << " " << modelAct << std::endl;
      }
   }
   return (float)RE/(float)batchsize_;
}

float SigmoidLayer :: freeEnergy_contibution() {
   /*expandBiases();
   
   //Computing x = W v + b
   gsl_matrix_float *x_j = gsl_matrix_float_alloc(up->nodenum_, up->batchsize_);
   gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1, weights_, activations_, 0, x_j);
   gsl_matrix_float_add(x_j, batchbiases_);
   
   gsl_vector_float *vBiasTerm = gsl_vector_float_alloc(batchsize_);
   gsl_vector_float *identity = gsl_vector_float_alloc(nodenum_);
   gsl_vector_float_set_all(identity, 1);
   gsl_vector_float *hterm = gsl_vector_float_alloc(batchsize_);
   
   //Compute v.a
   gsl_blas_sgemv(CblasTrans, 1, activations_,biases_, 0, vBiasTerm);
   
   // x_j -> softplus x_j
   for (int i = 0; i < up->nodenum_; ++i)
      for (int j = 0; j < up->batchsize_; ++j){
         float val = softplus(gsl_matrix_float_get(x_j, i, j));
         gsl_matrix_float_set(x_j, i, j, val);
      }
   // Then get x.hterm
   gsl_blas_sgemv(CblasNoTrans, 1, x_j, identity, 0, hterm);
   
   // add hterm and vbias term
   gsl_vector_float_add(hterm, vBiasTerm);
   
   freeEnergy_ = -gsl_stats_float_mean(hterm->data, hterm->stride, hterm->size);
   
   //Free memory
   gsl_vector_float_free(vBiasTerm);
   gsl_vector_float_free(hterm);
   gsl_matrix_float_free(x_j);
   gsl_vector_float_free(identity);*/
   return 0;
}

//The input needs to be shaped depending on the type of visible layer.
void SigmoidLayer::shapeInput(Input_t* input){
   float min, max;
   gsl_matrix_float_minmax(input, &min, &max);
   gsl_matrix_float_add_constant(input, -min);
   gsl_matrix_float_scale(input, (float)1/((float)(max-min)));
}