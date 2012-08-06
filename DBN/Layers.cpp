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
   means_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   activations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   
   up = NULL;
   
   weights_ = NULL;
   weightsT_ = weights_;
   batchbiases_ = gsl_matrix_float_alloc(nodenum_, batchsize_);
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
   gsl_matrix_float_free(means_);
   gsl_matrix_float_free(batchbiases_);
   
   batchsize_ = batchsize;
   
   activations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   preactivations_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   means_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
   batchbiases_ = gsl_matrix_float_calloc(nodenum_, batchsize_);
}
