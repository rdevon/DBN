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
   
   //Init stuff for layers
   Layer *top = rbm->top;
   Layer *bot = rbm->bot;
   top->on = true;
   bot->on = true;
   top->frozen = false;
   bot->frozen = false;
   //Make batch over batch size for matrix ops
   top->makeBatch(batchsize_);
   bot->makeBatch(batchsize_);
   top->preactivator = new PreActivator(UPFLAG, bot);
   bot->preactivator = new PreActivator(DOWNFLAG, top);
   
   //To randomize the input... don't want to modify or copy it *TODO*
   gsl_ran_shuffle(r, input->data, input->size1, input->size2 * sizeof(float));
   
   //Init stuff for sparse cost
   /*gsl_vector_float_free(oldnormterm_);
    oldnormterm_ = NULL;
    */
   
   std::cout << "Teaching RBM with input" << std::endl << "RBM dimensions: " << bot->nodenum_ << "x" << top->nodenum_ << std::endl << "Learning rate: " << learningRate_ << std::endl << "K: " << k_ << std::endl << "Batch Size: " << batchsize_ << std::endl;
   
   //Loop through the input batches
   for(int i = 0; i < (input->size1)/batchsize_; ++i){
      
      //-------First copy the batch onto the activation layers
      
      gsl_matrix_float_view inputbatch = gsl_matrix_float_submatrix(input, i, 0, batchsize_, input->size2);
      gsl_matrix_float_transpose_memcpy(bot->activations_, &(inputbatch.matrix));
      gsl_matrix_float_memcpy(v1, bot->activations_);
      
      //-----Gibbs sample visibible, hidden, then visible k-1 times
      
      top->activate(ACTIVATIONS);
      
      gsl_matrix_float_memcpy(h1, top->activations_);
      gsl_matrix_float_memcpy(Q1, top->probabilities_);
      
      for(int g = 0; g < k_; ++g) rbm->gibbs_VH(ACTIVATIONS);
      
      vk = bot->activations_;
      hk = top->activations_;
      Qk = top->probabilities_;
      
      //------Caclulating updates
      
      // W update = h1 v1^T - Qk vk^T
      gsl_blas_sgemm(CblasNoTrans, CblasTrans , 1, h1, v1, 0, WUpdate);
      gsl_blas_sgemm(CblasNoTrans, CblasTrans , -1, Qk, vk, 1, WUpdate);
      
      // top bias update = (h1 - Qk).I
      gsl_matrix_float_sub(h1, Qk);
      gsl_blas_sgemv(CblasNoTrans, 1, h1, identity, 0, topBiasUpdate);
      
      // bottom bias update = (v1 - vk).I
      gsl_matrix_float_sub(v1, vk);
      gsl_blas_sgemv(CblasNoTrans, 1, v1, identity, 0, botBiasUpdate);

      //Decay weights by cost
      gsl_matrix_float_memcpy(costterm, bot->weights_);
      gsl_matrix_float_scale(costterm, -weightcost_);
      gsl_matrix_float_add(WUpdate, costterm);
      
      //Calculate the normalizing term and add it
      
      /*newnormterm_ = gsl_vector_float_alloc(rbm->top->nodenum_);
      gsl_blas_sgemv(CblasNoTrans, 1, rbm->top->means_, identity, 0, newnormterm_);
      if (oldnormterm_ == NULL) oldnormterm_ = newnormterm_;
      else {
         gsl_vector_float_scale(newnormterm_, (float)(1-lambda_)/(float)rbm->top->batchsize_);
         gsl_vector_float_scale(oldnormterm_, lambda_);
         gsl_vector_float_add(newnormterm_, oldnormterm_);
         gsl_vector_float_free(oldnormterm_);
         oldnormterm_ = newnormterm_;
         gsl_vector_float_memcpy(sparsepenalty, newnormterm_);
         gsl_vector_float_add_constant(sparsepenalty, -p_);
         gsl_vector_float_scale(sparsepenalty, sparsitycost_);
         gsl_vector_float_sub(rbm->top->biases_, sparsepenalty);
         for (int i = 0; i < rbm->bot->nodenum_; ++i) gsl_matrix_float_set_col(wsparsepenalty, i, sparsepenalty);
         gsl_matrix_float_sub(rbm->bot->weights_, wsparsepenalty);
      }*/
      
      //Scale by learning rate
      gsl_matrix_float_scale(WUpdate, learningRate_/batchsize_);
      gsl_vector_float_scale(topBiasUpdate, learningRate_/batchsize_);
      gsl_vector_float_scale(botBiasUpdate, learningRate_/batchsize_);
      
      //update parameters
      gsl_matrix_float_add(bot->weights_, WUpdate);
      gsl_vector_float_add(top->biases_, topBiasUpdate);
      gsl_vector_float_add(bot->biases_, botBiasUpdate);
      
      //Visualization of the weights
      if ((i*batchsize_)%(input->size1/100) == 0){
         viz->clear();
         std::cout << i*batchsize_ << " ";
         /*for (int j = 0; j < fmin(viz->across*viz->down, rbm->bot->weights_->size1); ++j) {
            gsl_matrix_float_get_row(forvizvec, rbm->bot->weights_, j);
            viz->add(forvizvec);
         }*/
         for (int j = 0; j<20; ++j) {
            /*gsl_matrix_float_get_row(forvizvec, input, j);
            viz->add(forvizvec);*/                                ///Uncomment these if you want to see the input vectors
            gsl_matrix_float_get_row(forvizvec, rbm->bot->weights_, j);
            viz->add(forvizvec);
         }
         //update opengl window
         viz->updateViz();
         
      }
   }
   viz->plot();
   rbm->getReconstructionError(input);
   
}