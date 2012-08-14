//
//  Layers.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/18/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "Layers.h"

Connection::Connection(Layer *bot, Layer *top) :  bot_(bot), top_(top) {
   initializeWeights();
   mat_update = gsl_matrix_float_calloc(top_->nodenum_, bot_->nodenum_);
   stat1 = gsl_matrix_float_alloc(top_->nodenum_, bot_->nodenum_);
   stat2 = gsl_matrix_float_alloc(top_->nodenum_, bot_->nodenum_);
}

void Connection::update(ContrastiveDivergence* teacher){
   
   gsl_matrix_float *weight_update = mat_update;
   
   float learning_rate = teacher->learningRate_/((float)teacher->batchsize_*(float)teacher->batchsize_);
   gsl_blas_sgemm(CblasNoTrans, CblasTrans , learning_rate, teacher->top_pos_stats_, teacher->bot_pos_stats_, teacher->momentum_, weight_update);
   gsl_blas_sgemm(CblasNoTrans, CblasTrans , -learning_rate, teacher->top_neg_stats_, teacher->bot_neg_stats_, 1, weight_update);
   
   gsl_matrix_float_add(weights_, weight_update);
   
   top_->update(teacher);
   bot_->update(teacher);
}

void Connection::initializeWeights(){
   
   // Set weights to initial normal distribution with 0 mean and 0.01 variance.  This is suggested by Hinton but different from tutorial which uses
   // initial uniform distribution between -+4 sqrt(6/(h_nodes+vnodes))
   
   weights_ = gsl_matrix_float_alloc(top_->nodenum_, bot_->nodenum_);
   for (int i = 0; i < top_->nodenum_; ++i)
      for (int j = 0; j < bot_->nodenum_; ++j)
         gsl_matrix_float_set(weights_, i, j, (float)gsl_ran_gaussian(r, 0.01));
}

void Connection::prop(Up_flag_t up, Sample_flag_t s){
   CBLAS_TRANSPOSE_t transFlag;
   Layer *signal_layer;
   Layer *layer;
   gsl_matrix_float *signal;
   if (up == UPFLAG){
      layer = top_;
      signal_layer = bot_;
      transFlag = CblasNoTrans;
   }
   else {
      layer = bot_;
      signal_layer = top_;
      transFlag = CblasTrans;
   }
   
   if (s == SAMPLE) signal = signal_layer->samples_;
   else signal = signal_layer->expectations_;
   
   if (layer->frozen == false){
      layer->expandBiases();
      gsl_blas_sgemm(transFlag, CblasNoTrans, 1, weights_, signal, 1, layer->activations_);
      gsl_matrix_float_add(layer->activations_, layer->batchbiases_);
      layer->expectation_up_to_date = false;
      layer->sample_up_to_date = false;
      layer->learning_up_to_date = false;
   }
}

void Connection::initprop(Up_flag_t up){
   if (up == UPFLAG) gsl_matrix_float_set_all(top_->activations_, 0);
   else gsl_matrix_float_set_all(bot_->activations_, 0);
}

void Connection::makeBatch(int batchsize) {
   top_->makeBatch(batchsize);
   bot_->makeBatch(batchsize);
}

void Connection::expandBiases(){
   top_->expandBiases();
   bot_->expandBiases();
}

void Activator::activate(){
   c1_->initprop(up_flag_);
   //c2_->initprop(up_flag_);
   //c3_->initprop(up_flag_);
   c1_->prop(up_flag_, s_flag_);
   //c2_->prop(up_flag_);
   //c3_->prop(up_flag_);
}

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