//
//  Connections.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/15/12.
//
//

#include "Connections.h"

Connection::Connection(Layer *bot, Layer *top) : Learner(), bot_(bot), top_(top) {
   initializeWeights();
   mat_update = gsl_matrix_float_calloc(top_->nodenum_, bot_->nodenum_);
}

void Connection::update(ContrastiveDivergence* teacher){
   
   gsl_matrix_float *weight_update = mat_update;
   
   float learning_rate = teacher->learningRate_/((float)teacher->batchsize_);
   //learning_rate/=(float)teacher->batchsize_;
   gsl_blas_sgemm(CblasNoTrans, CblasTrans , learning_rate, stat1, stat2, teacher->momentum_, weight_update);
   gsl_blas_sgemm(CblasNoTrans, CblasTrans , -learning_rate, stat3, stat4, 1, weight_update);
   
   gsl_matrix_float *weightdecay = gsl_matrix_float_alloc(weights_->size1, weights_->size2);
   gsl_matrix_float_memcpy(weightdecay, weights_);
   gsl_matrix_float_scale(weightdecay, decay_);
   gsl_matrix_float_sub(weight_update, weightdecay);
   
   gsl_matrix_float_add(weights_, weight_update);
   
   top_->update(teacher);
   bot_->update(teacher);
   
   gsl_matrix_float_free(weightdecay);
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
   
   //gsl_matrix_float *eff_weights = gsl_matrix_float_alloc(weights_->size1, weights_->size2);
   
   
   if (layer->frozen == false){
      layer->expandBiases();
      gsl_blas_sgemm(transFlag, CblasNoTrans, 1, weights_, signal, 1, layer->activations_);
      gsl_matrix_float_add(layer->activations_, layer->batchbiases_);
      layer->expectation_up_to_date = false;
      layer->sample_up_to_date = false;
      layer->learning_up_to_date = false;
   }
   else {
      layer->expectation_up_to_date = true;
      layer->sample_up_to_date = true;
      layer->learning_up_to_date = true;
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

void Connection::catch_stats(Stat_flag_t s){
   stat1 = top_->stat1;
   stat2 = bot_->stat1;
   stat3 = top_->stat2;
   stat4 = bot_->stat2;
   bot_->catch_stats(s, SAMPLE);
   if (s == NEG) top_->catch_stats(s, NOSAMPLE);
   else top_->catch_stats(s, SAMPLE);
}

void Activator::activate(){
   c1_->initprop(up_flag_);
   if (c2_ != NULL) c2_->initprop(up_flag_);
   //c3_->initprop(up_flag_);
   c1_->prop(up_flag_, s_flag_);
   if (c2_ != NULL) c2_->prop(up_flag_, s_flag_);
   //c3_->prop(up_flag_);
}
