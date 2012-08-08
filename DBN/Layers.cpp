//
//  Layers.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/18/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "Layers.h"

Layer::Layer(int nodenum){
   nodenum_ = nodenum;
   batchsize_ = 1;      //Initialize to 1 but we'll redo these as the batchsize changes
   
   preactivations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   probabilities_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   activations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   
   up = NULL;
   
   weights_ = NULL;
   weightsT_ = weights_;
   batchbiases_ = gsl_matrix_float_alloc(nodenum_, batchsize_);
   frozen = false;
   on = true;
}

void Layer::initializeWeights(){
   
   // Set weights to initial normal distribution with 0 mean and 0.01 variance.  This is suggested by Hinton but different from tutorial which uses
   // initial uniform distribution between -+4 sqrt(6/(h_nodes+vnodes))
   
   weights_ = gsl_matrix_float_alloc(up->nodenum_, nodenum_);
   for (int i = 0; i < up->nodenum_; ++i)
      for (int j = 0; j < nodenum_; ++j)
         gsl_matrix_float_set(weights_, i, j, (float)gsl_ran_gaussian(r, 0.01));
}

//Default is sigmoid
void Layer::addLayer(int newnodes){
   if (up==NULL){
      up = new SigmoidLayer(newnodes);
      initializeWeights();
   }
   else up->addLayer(newnodes);
}

void Layer::addLayer(Layer *newlayer){
   if (up==NULL){
      up = newlayer;
      initializeWeights();
   }
   else up->addLayer(newlayer);
}

Layer *Layer::getTop(){
   if (up==NULL) return this;
   else return up->getTop();
}

void Layer::makeBatch(int batchsize){
   
   //For batch processing.  
   gsl_matrix_float_free(activations_);
   gsl_matrix_float_free(preactivations_);
   gsl_matrix_float_free(probabilities_);
   gsl_matrix_float_free(batchbiases_);
   
   batchsize_ = batchsize;
   
   activations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   preactivations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   probabilities_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   batchbiases_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
}

void Layer::activate(Activation_flag_t act){
   if (not(frozen) && on){
      preactivator->preactivate(this);
      
      setProbs();
      
      //Sample if sample flag on
      if (act == ACTIVATIONS){
         for (int i = 0; i < nodenum_; ++i){
            for (int j = 0; j < batchsize_; ++j){
               float u = gsl_rng_uniform(r);
               float act = (float)(gsl_matrix_float_get(probabilities_, i, j) > u);
               gsl_matrix_float_set(activations_, i, j, act);
            }
         }
      }
      else gsl_matrix_float_memcpy(activations_, probabilities_);
   }
   if (not(on)) {
      gsl_matrix_float_set_all(preactivations_, 0);
      gsl_matrix_float_set_all(probabilities_, 0);
      gsl_matrix_float_set_all(activations_, 0);
   }
}

void PreActivator::preactivate(Layer *layer){
   CBLAS_TRANSPOSE_t transFlag;
   gsl_matrix_float *weights;
   //Check up or down flag to set transpose flag
   if (up_) transFlag = CblasNoTrans;
   else transFlag = CblasTrans;
   
   if (sL1_->on){
      //Expand the biases into a matrix for matrix operation
      for (int j = 0; j < layer->batchsize_; ++j) gsl_matrix_float_set_col(layer->batchbiases_, j, layer->biases_);
      
      //Check flag for up or down activation
      if (up_) weights = sL1_->weights_;
      else weights = layer->weights_;
      
      //Compute x = W^T v + b or W v + b depending on up or down activation
      
      gsl_blas_sgemm(transFlag, CblasNoTrans, 1, weights, sL1_->activations_, 0, layer->preactivations_);
      gsl_matrix_float_add(layer->preactivations_, layer->batchbiases_);
   }
   
   // Add the other preactivations if they exist
   if (sL2_ != NULL && sL2_->on){
      if (up_) weights = sL2_->weights_;
      gsl_blas_sgemm(transFlag, CblasNoTrans, 1, weights, sL2_->activations_, 1, layer->preactivations_);
      gsl_matrix_float_add(layer->preactivations_, layer->batchbiases_);
   }
   
   if (sL3_ != NULL && sL3_->on){
      if (up_) weights = sL3_->weights_;
      gsl_blas_sgemm(transFlag, CblasNoTrans, 1, weights, sL3_->activations_, 1, layer->preactivations_);
      gsl_matrix_float_add(layer->preactivations_, layer->batchbiases_);
   }
   
}
