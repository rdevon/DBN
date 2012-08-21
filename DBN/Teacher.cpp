//
//  Teacher.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/10/12.
//
//

#include "Teacher.h"
#include "RBM.h"
#include "Viz.h"

ContrastiveDivergence::ContrastiveDivergence(RBM *rbm, DataSet *data, float learningRate, float weightcost, float momentum, int k, float p, float lambda, float sparsitycost, int batchsize) : rbm_(rbm), data_(data), learningRate_(learningRate), weightcost_(weightcost), momentum_(momentum), k_(k), p_(p), lambda_(lambda), sparsitycost_(sparsitycost), batchsize_(batchsize)
{
   
   identity = gsl_vector_float_alloc(batchsize_);
   gsl_vector_float_set_all(identity, 1);
   forvizvec = gsl_vector_float_alloc(rbm_->c1_->bot_->nodenum_);
   viz_ = new Visualizer(rbm_->c1_->top_->nodenum_ ,data_);
   // Make batch over batch size for matrix ops
   rbm_->makeBatch(batchsize_);
}

void ContrastiveDivergence::getStats(){
   Layer *top = rbm_->c1_->top_;
   
   // Activate the top layer, get the expectations, and sample.
   rbm_->up_act_->activate();
   top->getExpectations();
   top->sample();

   // Positive stats
   rbm_->catch_stats(POS);
   
   // Gibbs VH sample k times.
   for(int g = 0; g < k_; ++g) rbm_->gibbs_VH();
   
   // Negative stats.
   rbm_->catch_stats(NEG);
   
}

void ContrastiveDivergence::run(){
   // Turn the sample flags on
   rbm_->up_act_->s_flag_ = SAMPLE;
   rbm_->down_act_->s_flag_ = SAMPLE;
   Layer *bot = rbm_->c1_->bot_;
   Layer *bot2 = rbm_->c2_->bot_;
   gsl_matrix_float *input = data_->train;
   gsl_matrix_float *stim = data_->stim;
   
   gsl_rng *r_temp = gsl_rng_alloc(gsl_rng_rand48);
   gsl_rng_memcpy(r_temp, r);
   
   gsl_ran_shuffle(r_temp, input->data, input->size1, input->size2*sizeof(float));
   gsl_ran_shuffle(r, stim->data, stim->size1, stim->size2*sizeof(float));
   gsl_rng_free(r_temp);
   
   // Get dimensions
   float topdim, botdim;
   rbm_->get_dims(&topdim, &botdim);
   
   std::cout << "Teaching RBM with input" << std::endl << "RBM dimensions: " << botdim << "x" << topdim << std::endl << "Learning rate: " << learningRate_ << std::endl << "K: " << k_ << std::endl << "Batch Size: " << batchsize_ << std::endl;
   
   // Loop through the input
   for(int i = 0; i < (input->size1)/batchsize_; ++i){
      
      // Make a batch of the input and perform CD
      gsl_matrix_float_view inputbatch = gsl_matrix_float_submatrix(input, i, 0, batchsize_, input->size2);
      gsl_matrix_float_view stimbatch = gsl_matrix_float_submatrix(stim, i, 0, batchsize_, stim->size2);
      
      // Copy the input batch onto the samples of the visible layer
      gsl_matrix_float_transpose_memcpy(bot->samples_, &(inputbatch.matrix));
      gsl_matrix_float_transpose_memcpy(bot2->samples_, &(stimbatch.matrix));
      
      getStats();
      
      // Update the parameters.
      rbm_->update(this);
      
      // And monitor
      if ((i*batchsize_)%(input->size1/100) == 0) monitor(i*batchsize_);
   }
   //rbm_->getReconstructionCost(input);
}

void ContrastiveDivergence::monitor(int i){
   //std::cout << i << " ";
   viz_->clear();
   //viz_->add(rbm_->c1_->bot_->vec_update2);
   for (int j = 0; j<rbm_->c1_->weights_->size1; ++j) {
      /*gsl_matrix_float_get_row(forvizvec, data_->train, j);
      viz_->add(forvizvec); */     ///Uncomment these if you want to see the input vectors
      gsl_matrix_float_get_row(forvizvec, rbm_->c1_->weights_, j);
      viz_->add(forvizvec);
   }
   //update opengl window
   viz_->updateViz();
   viz_->plot();
}