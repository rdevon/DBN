//
//  Learner.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "RBM.h"
#include "gsl/gsl_rng.h"

void CD::teach(RBM *rbm, Input_t *input){
   
   rbm->bot->shapeInput(input);
   
   rbm->activator.flag_ = ACTIVATIONS;
   
   //Make batch over batch size for matrix ops
   rbm->top->makeBatch(batchsize_);
   rbm->bot->makeBatch(batchsize_);
   
   //To randomize the input... don't want to modify or copy it *TODO*
   
   gsl_vector_float *forvizvec = gsl_vector_float_alloc(rbm->bot->nodenum_);
   
   gsl_matrix_float *h1 = gsl_matrix_float_alloc(rbm->top->nodenum_, rbm->top->batchsize_);
   gsl_matrix_float *hk; //Don't have to allocate because this is the last update
   
   gsl_matrix_float *v1 = gsl_matrix_float_alloc(rbm->bot->nodenum_, batchsize_); //the sample
   gsl_matrix_float *vk;
   
   gsl_matrix_float *Q1 = gsl_matrix_float_alloc(rbm->top->nodenum_, batchsize_);
   gsl_matrix_float *Qk;
   
   //Initializes the update vectors and matrix to 0.  
   gsl_matrix_float *WUpdate = gsl_matrix_float_calloc(rbm->top->nodenum_, rbm->bot->nodenum_);
   gsl_vector_float *topBiasUpdate = gsl_vector_float_alloc(rbm->top->nodenum_);
   gsl_vector_float *botBiasUpdate = gsl_vector_float_alloc(rbm->bot->nodenum_);
   gsl_vector_float_set_all(topBiasUpdate, 1);
   gsl_vector_float_set_all(botBiasUpdate, 1);
   gsl_vector_float *identity = gsl_vector_float_alloc(batchsize_);
   gsl_vector_float_set_all(identity, 1);
   
   std::cout << "Teaching RBM with input" << std::endl << "RBM dimensions: " << rbm->bot->nodenum_ << "x" << rbm->top->nodenum_ << std::endl << "Learning rate: " << learningRate_ << std::endl << "K: " << k_ << std::endl << "Batch Size: " << batchsize_ << std::endl;
   
   for(int i = 0; i < (input->size1)/batchsize_; ++i){
      std::cout << std::flush;
      
      //First copy the batch onto the activation layers
      gsl_matrix_float_view inputbatch = gsl_matrix_float_submatrix(input, i, 0, batchsize_, input->size2);
      gsl_matrix_float_transpose_memcpy(rbm->bot->activations_, &(inputbatch.matrix));
      gsl_matrix_float_memcpy(v1, rbm->bot->activations_);
      
      //Gibbs sample visibible, hidden, then visible k-1 times
      rbm->top->activate((rbm->activator), rbm->bot, UPFLAG);
      gsl_matrix_float_memcpy(h1, rbm->top->activations_);
      gsl_matrix_float_memcpy(Q1, rbm->top->means_);
      
      for(int g = 0; g < k_; ++g) rbm->gibbs_VH();
      
      vk = rbm->bot->activations_;
      hk = rbm->top->activations_;
      Qk = rbm->top->means_;
      
      // W update = h1 v1^T - Qk vk^T
      gsl_blas_sgemm(CblasNoTrans, CblasTrans , 1, h1, v1, 0, WUpdate);
      gsl_blas_sgemm(CblasNoTrans, CblasTrans , -1, Qk, vk, 1, WUpdate);
      
      // top bias update = (h1 - Qk).I
      gsl_matrix_float_scale(Qk, -1);
      gsl_matrix_float_add(h1, Qk);
            
      gsl_blas_sgemv(CblasNoTrans, 1, h1, identity, 0, topBiasUpdate);
      // bottom bias update = (v1 - vk).I
      gsl_matrix_float_scale(vk, -1);
      gsl_matrix_float_add(v1, vk);
      gsl_blas_sgemv(CblasNoTrans, 1, v1, identity, 0, botBiasUpdate);
      
      //update
      gsl_matrix_float_scale(WUpdate, learningRate_/batchsize_);
      gsl_vector_float_scale(topBiasUpdate, learningRate_/batchsize_);
      gsl_vector_float_scale(botBiasUpdate, learningRate_/batchsize_);
      
      gsl_matrix_float_add(rbm->bot->weights_, WUpdate);
      gsl_vector_float_add(rbm->top->biases_, topBiasUpdate);
      gsl_vector_float_add(rbm->bot->biases_, botBiasUpdate);
      
      //Visualization of the weights
      if ((i*batchsize_)%(input->size1/100) == 0){
         viz->clear();
         std::cout << i*batchsize_ << " ";
         for (int j = 0; j < fmin(viz->across*viz->down, rbm->bot->weights_->size1); ++j) {
            gsl_matrix_float_get_row(forvizvec, rbm->bot->weights_, j);
            viz->add(forvizvec);
         }
         //update opengl window
         viz->updateViz();
         viz->plot();
         
      }
   }
   rbm->getReconstructionError(input);
   
   //Free memory of all tempory objects used for learning.
   gsl_matrix_float_free(v1);
   gsl_matrix_float_free(Q1);
   gsl_matrix_float_free(WUpdate);
   gsl_vector_float_free(topBiasUpdate);
   gsl_vector_float_free(botBiasUpdate);
   gsl_vector_float_free(identity);
   
}