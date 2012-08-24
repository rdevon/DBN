//
//  RBM.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "RBM.h"

RBM::RBM(Connection* c1) : c1_(c1), c2_(NULL), reconstructionCost_(0) {
   up_act_ = new Activator(UPFLAG, c1_);
   down_act_ = new Activator(DOWNFLAG, c1_);
}

RBM::RBM(Connection* c1, Connection* c2) : c1_(c1), c2_(c2), reconstructionCost_(0){
   up_act_ = new Activator(UPFLAG, c1_, c2_);
   down_act_ = new Activator(DOWNFLAG, c1_, c2_);
}

void RBM::getFreeEnergy(){
}

void RBM::gibbs_HV(){
   up_act_->activate();
   c1_->top_->getExpectations();
   c1_->top_->sample();
   if (c2_ != NULL){
      c2_->top_->getExpectations();
      c2_->top_->sample();
   }
   
   down_act_->activate();
   c1_->bot_->getExpectations();
   c1_->bot_->sample();
   if (c2_ != NULL){
      c2_->bot_->getExpectations();
      c2_->bot_->sample();
   }
   
}

void RBM::gibbs_VH(){
   down_act_->activate();
   c1_->bot_->getExpectations();
   c1_->bot_->sample();
   if (c2_ != NULL){
      c2_->bot_->getExpectations();
      c2_->bot_->sample();
   }
   
   up_act_->activate();
   c1_->top_->getExpectations();
   c1_->top_->sample();
   if (c2_ != NULL){
      c2_->top_->getExpectations();
      c2_->top_->sample();
   }
}

void RBM::makeBatch(int batchsize){
   c1_->makeBatch(batchsize);
   c2_->makeBatch(batchsize);
}

void RBM::expandBiases(){
   c1_->expandBiases();
   c2_->expandBiases();
}

void RBM::get_dims(float *topdim, float *botdim){
   *topdim = c1_->top_->nodenum_;
   *botdim = c1_->bot_->nodenum_;
}

void RBM::update(ContrastiveDivergence *cd){
   c1_->update(cd);
   c2_->update(cd);
}

void RBM::getReconstructionCost(Input_t *input){
   float oldbatch = c1_->bot_->batchsize_;
   Layer *bot = c1_->bot_;
   
   //Make batch over entire input for matrix ops
   makeBatch((int)input->size1);
   
   //Enter the entire input as the activations and means.  for some reason gsl_matrix_transpose_memcopy isn't working here.
   gsl_matrix_float *dataMat = gsl_matrix_float_calloc(input->size2, input->size1);
   gsl_matrix_float *modelMat;
   
   gsl_matrix_float_transpose_memcpy(dataMat, input);
   
   gsl_matrix_float_memcpy(bot->samples_, dataMat);
   gsl_matrix_float_memcpy(bot->expectations_, dataMat);
   
   up_act_->s_flag_ = NOSAMPLE;
   down_act_->s_flag_ = NOSAMPLE;
   
   //gibbs once over the whole matrix
   gibbs_HV();
   
   modelMat = bot->expectations_;
   
   // calculate the reconstruction cost as per the visible layer type
   reconstructionCost_ = bot->reconstructionCost(dataMat, modelMat);
   
   std::cout << "Reconstruction cost: " << reconstructionCost_ << std::endl;
   
   gsl_matrix_float_free(dataMat);
   
   // Revert to the old batchsize.
   makeBatch(oldbatch);
}

void RBM::sample(DataSet *data, Visualizer *viz){
   Layer *top = c1_->top_;
   Layer *bot = c1_->bot_;
   gsl_matrix_float *input;
   if (data->test == NULL) input = data->train;
   else input = data->test;
   
   up_act_->s_flag_ = NOSAMPLE;
   down_act_->s_flag_ = NOSAMPLE;
   
   std::cout << "Sampling RBM" << std::endl;
   
   viz->clear();
   
   top->makeBatch(viz->across*viz->down);
   bot->makeBatch(viz->across*viz->down);
   
   gsl_vector_float *samples = gsl_vector_float_alloc(input->size2);
   for (int j = 0; j < viz->across*viz->down; ++j){ // As many samples as will fit in the viz.
      int u = (int)gsl_rng_uniform_int(r, input->size1);
      gsl_matrix_float_get_row(samples, input, u);
      gsl_matrix_float_set_col(bot->samples_, j, samples);
      gsl_matrix_float_set_col(bot->expectations_, j, samples);
   }
   
   for (int epoch = 0; epoch < 1000; ++epoch) {
      gibbs_HV();
   }
   for (int j = 0; j < viz->across*viz->down; ++j){
      gsl_matrix_float_get_col(samples, bot->samples_, j);
      viz->add(samples);
   }
   
   viz->plot();
   gsl_vector_float_free(samples);
}

void RBM::load_DS(DataSet *ds1, DataSet *ds2 = NULL){
   ds1_ = ds1;
   ds2_ = ds2;
   c1_->bot_->shapeInput(ds1_);
   if (c2_ != NULL) c2_->bot_->shapeInput(ds2_);
}

void RBM::load_input_batch(int index){
   gsl_matrix_float *ds1 = ds1_->train;
   int batchsize = c1_->bot_->batchsize_;
   
   // Make a batch of the input and perform CD
   gsl_matrix_float_view inputbatch1 = gsl_matrix_float_submatrix(ds1, index, 0, batchsize, ds1->size2);
   
   // Copy the input batch onto the samples of the visible layer
   gsl_matrix_float_transpose_memcpy(c1_->bot_->samples_, &(inputbatch1.matrix));
   
   if (ds2_ != NULL){
      gsl_matrix_float *ds2 = ds2_->train;
      gsl_matrix_float_view inputbatch2 = gsl_matrix_float_submatrix(ds2, index, 0, batchsize, ds2->size2);
      gsl_matrix_float_transpose_memcpy(c2_->bot_->samples_, &(inputbatch2.matrix));
   }
}

void RBM::init_DS(){
   gsl_matrix_float *ds1 = ds1_->train;
   
   gsl_rng *r_temp = gsl_rng_alloc(gsl_rng_rand48);
   gsl_rng_memcpy(r_temp, r);
   
   gsl_ran_shuffle(r_temp, ds1->data, ds1->size1, ds1->size2*sizeof(float));
   
   if (ds2_ != NULL) {
      gsl_matrix_float *ds2 = ds2_->train;
      gsl_ran_shuffle(r, ds2->data, ds2->size1, ds2->size2*sizeof(float));
   }
   
   gsl_rng_free(r_temp);
}

void RBM::visualize(float st1, float st2){
   Visualizer viz(16, ds1_);
   makeBatch(1);
   gsl_matrix_float_set(c2_->bot_->samples_, 0, 0, st1);
   gsl_matrix_float_set(c2_->bot_->samples_, 1, 0, st2);
   c2_->bot_->frozen = true;
   gsl_matrix_float_set_all(c1_->bot_->samples_, 0);
   
   gsl_vector_float *for_viz = gsl_vector_float_calloc(c1_->bot_->nodenum_);
   while (1){
      viz.clear();
      gibbs_HV();
      gsl_matrix_float_get_col(for_viz, c1_->bot_->expectations_, 0);
      viz.add(for_viz);
      viz.updateViz();
   }
   gsl_vector_float_free(for_viz);
}