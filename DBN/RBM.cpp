//
//  RBM.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "RBM.h"

RBM::RBM(Layer* layer) : bot(layer), top(layer->up){}

void RBM::getFreeEnergy(){
   bot->getFreeEnergy();
   freeEnergy_ = bot->freeEnergy_;
   activator = Activator();
   activator.flag_ = ACTIVATIONS;
}

void RBM::gibbs_HV(){
   top->activate(activator, bot, UPFLAG);
   bot->activate(activator, top, DOWNFLAG);
}

void RBM::gibbs_VH(){
   bot->activate(activator, top, DOWNFLAG);
   top->activate(activator, bot, UPFLAG);
}

void RBM::getReconstructionError(Input_t *input){
   //Make batch over entire input for matrix ops
   top->makeBatch((int)input->size1);
   bot->makeBatch((int)input->size1);
   
   //set flag to probs as this is less noisy and best for reconstructions
   activator.flag_ = PROBABILITIES;
   
   std::cout << std::endl << "Calculating Reconstruction Error" << std::endl;
   double RE = 0;
   
   //Enter the entire input as the activations and means
   gsl_matrix_float *dataMat = gsl_matrix_float_alloc(input->size2, input->size1), *modelMat;
   gsl_matrix_float_transpose_memcpy(dataMat, input);
   gsl_matrix_float_memcpy(bot->activations_, dataMat);
   gsl_matrix_float_memcpy(bot->means_, dataMat);
   
   //gibbs once over the whole matrix
   gibbs_HV();
   
   modelMat = bot->activations_;
   
   //Hmmmm worth doing batches?  Dunno.  Since I have to go through by hand and modify every entry.  Maybe the compiler can handle this
   for (int i = 0; i < bot->nodenum_; ++i){
      for (int j = 0; j < bot->batchsize_; ++j){
         //We use the reop operator here in case layers have weird behavior that we need to catch when caculating the reconstruction error.
         //double dataAct = bot->reop(gsl_vector_float_get(dataVec, i));
         //double modelAct = bot->reop(gsl_vector_float_get(modelVec, i));
         double dataAct = gsl_matrix_float_get(dataMat, i, j);
         double modelAct = gsl_matrix_float_get(modelMat, i, j);
         RE += dataAct * log(modelAct) + (1-dataAct)*log(1- modelAct); //Cost really.
         //if ((modelAct >=1)||(modelAct <= 0))std::cout << (1-dataAct)*log(1- modelAct) << " " << dataAct * log(modelAct) << " " << dataAct << " " << modelAct << std::endl;
      }
   }
   reconstructionError_ = (float)RE/(float)input->size1;
   gsl_matrix_float_free(dataMat);
}

void RBM::sample(DataSet *data, Visualizer *viz){
   gsl_matrix_float *input;
   if (data->test == NULL) input = data->train;
   else input = data->test;
   
   //For sampling, probabilities are best
   activator.flag_ = PROBABILITIES;
   
   std::cout << "Sampling RBM" << std::endl;
   
   viz->clear();
   
   top->makeBatch(viz->across*viz->down);
   bot->makeBatch(viz->across*viz->down);
   
   gsl_vector_float *samples = gsl_vector_float_alloc(input->size2);
   for (int j = 0; j < viz->across*viz->down; ++j){ // As many samples as will fit in the viz.
      int u = (int)gsl_rng_uniform_int(r, input->size1);
      gsl_matrix_float_get_row(samples, input, u);
      gsl_matrix_float_set_col(bot->activations_, j, samples);
      gsl_matrix_float_set_col(bot->means_, j, samples);
   }
      
   for (int epoch = 0; epoch < 1000; ++epoch) {
      gibbs_HV();
   }
   for (int j = 0; j < viz->across*viz->down; ++j){
      gsl_matrix_float_get_col(samples, bot->activations_, j);
      viz->add(samples);
   }
   
   viz->plot();
   gsl_vector_float_free(samples);
}

