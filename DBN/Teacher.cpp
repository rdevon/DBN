//
//  Teacher.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/10/12.
//
//

#include "Teacher.h"
#include "RBM.h"
#include "Connections.h"
#include "Viz.h"
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_permute.h>



ContrastiveDivergence::ContrastiveDivergence(RBM *rbm, float momentum, int k, float p, float lambda, float sparsitycost, int batchsize) : momentum_(momentum), k_(k), p_(p), lambda_(lambda), sparsitycost_(sparsitycost), batchsize_(batchsize)
{
   identity = gsl_vector_float_alloc(batchsize_);
   gsl_vector_float_set_all(identity, 1);
   forvizvec = gsl_vector_float_alloc(rbm->c1_->bot_->nodenum_);
   viz_ = new Visualizer(rbm->c1_->top_->nodenum_ ,rbm->ds1_);
   viz_->initViz();
   // Make batch over batch size for matrix ops
   rbm->makeBatch(batchsize_);
   viz_->scale = 8;
}

void ContrastiveDivergence::getStats(RBM *rbm){
   Layer *top = rbm->c1_->top_;
   
   // Activate the top layer, get the expectations, and sample.
   rbm->up_act_->activate();
   top->getExpectations();
   top->sample();

   // Positive stats
   rbm->catch_stats(POS);
   
   // Gibbs VH sample k times.
   for(int g = 0; g < k_; ++g) rbm->gibbs_VH();
   
   // Negative stats.
   rbm->catch_stats(NEG);
   
}

void ContrastiveDivergence::teachRBM(RBM *rbm){
   // Turn the sample flags on
   rbm->up_act_->s_flag_ = SAMPLE;
   rbm->down_act_->s_flag_ = SAMPLE;
   
   // Get dimensions
   float topdim, botdim;
   rbm->get_dims(&topdim, &botdim);
   
   std::cout << "Teaching RBM with input" << std::endl << "RBM dimensions: " << botdim << "x" << topdim << std::endl << "K: " << k_ << std::endl << "Batch Size: " << batchsize_ << std::endl;
   
   rbm->init_DS();
   
   // Loop through the input
   for(int i = 0; i < (rbm->ds1_->train->size1)/batchsize_; ++i){
      
      // Loads the input batches into each base layer
      rbm->load_input_batch(i);
      
      // Gets statistics by performing CD.
      getStats(rbm);
      
      // Update the parameters.
      rbm->update(this);
      
      // And monitor
      if ((i*batchsize_)%(rbm->ds1_->train->size1/10) == 0) monitor(rbm, i*batchsize_);
   }
   rbm->getReconstructionCost();
}

void ContrastiveDivergence::monitor(RBM *rbm, int i){
   gsl_matrix_float *weights = rbm->c1_->weights_;
   /*gsl_vector_float *sums = gsl_vector_float_alloc(rbm_->c1_->weights_->size1);
   for (int i = 0; i < weights->size1; ++i){
      float sum = 0;
      for (int j = 0; j < weights->size2; ++j)
         sum += pow(gsl_matrix_float_get(rbm_->c1_->weights_, i , j),2);
      gsl_vector_float_set(sums, i, sum);
   }
   gsl_permutation *p = gsl_permutation_alloc(rbm_->c1_->weights_->size1);
   gsl_sort_vector_float_index(p, sums);*/
   
   viz_->clear();
   for (int j = 0; j<weights->size1; ++j) {
      /*gsl_matrix_float_get_row(forvizvec, data_->train, j);
      viz_->add(forvizvec); */     ///Uncomment these if you want to see the input vectors
      //size_t index = p->data[j];
      //gsl_matrix_float_get_row(forvizvec, weights, index);
      gsl_matrix_float_get_row(forvizvec, weights, j);
      viz_->add(forvizvec);
   }
   //update opengl window
   viz_->updateViz();
   viz_->plot();
   /*gsl_vector_float_free(sums);
   gsl_permutation_free(p);*/
}