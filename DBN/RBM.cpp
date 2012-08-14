//
//  RBM.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "RBM.h"

RBM::RBM(Connection* c1) : c1_(c1), reconstructionCost_(0) {
   up_act_ = new Activator(UPFLAG, c1_);
   down_act_ = new Activator(DOWNFLAG, c1_);
}

void RBM::getFreeEnergy(){
}

void RBM::gibbs_HV(){
   up_act_->activate();
   c1_->top_->getExpectations();
   c1_->top_->sample();
   
   down_act_->activate();
   c1_->bot_->getExpectations();
   c1_->bot_->sample();
}

void RBM::gibbs_VH(){
   down_act_->activate();
   c1_->bot_->getExpectations();
   c1_->bot_->sample();
   
   up_act_->activate();
   c1_->top_->getExpectations();
   c1_->top_->sample();
}

void RBM::makeBatch(int batchsize){
   c1_->makeBatch(batchsize);
}

void RBM::expandBiases(){
   c1_->expandBiases();
}

void RBM::get_dims(float *topdim, float *botdim){
   *topdim = c1_->top_->nodenum_;
   *botdim = c1_->bot_->nodenum_;
}

void RBM::update(ContrastiveDivergence *cd){
   c1_->update(cd);
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

