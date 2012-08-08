//
//  RBM.h
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_RBM_h
#define DBN_RBM_h
#include "Layers.h"
#include "SupportFunctions.h"
#include "Viz.h"
#include "IO.h"

class RBM {
public:
   
   float freeEnergy_, reconstructionError_;
   
   Layer *top;
   Layer *bot;
   
   ~RBM(){}
   RBM(){}
   RBM(Layer*);
   RBM(Layer*,Layer*);
   
   void getFreeEnergy();
   void gibbs_HV(Activation_flag_t);
   void gibbs_VH(Activation_flag_t);
   void getReconstructionError(Input_t *input);
   void sample(DataSet *data, Visualizer *viz);
};


class Learner {
public:
   
   ~Learner(){}
   Learner(){}
   
   virtual void teach(RBM*, Input_t *input) = 0;
};

class CD : public Learner{
public:
   float learningRate_, weightcost_, sparsitycost_;
   int k_;
   int batchsize_;
   float p_, lambda_;
   
   gsl_matrix_float *h1, *hk, *v1, *vk, *Q1, *Qk, *WUpdate, *costterm, *wsparsepenalty;
   gsl_vector_float *topBiasUpdate, *botBiasUpdate, *sparsepenalty, *identity, *newnormterm_, *oldnormterm_, *forvizvec;
   
   Visualizer *viz;
   
   ~CD(){}
   CD(){}
   CD(float learningRate, float weightcost, int k, float p, float lambda, float sparsitycost, Visualizer *v, int batchsize) : learningRate_(learningRate), weightcost_(weightcost), k_(k), p_(p), lambda_(lambda), sparsitycost_(sparsitycost), viz(v), batchsize_(batchsize) {}
   
   void inittemps(RBM* rbm){
      
      forvizvec = gsl_vector_float_alloc(rbm->bot->nodenum_);
      h1 = gsl_matrix_float_alloc(rbm->top->nodenum_, batchsize_);
      //Don't have to allocate because this is the last update
      v1 = gsl_matrix_float_alloc(rbm->bot->nodenum_, batchsize_); //the sample
      Q1 = gsl_matrix_float_alloc(rbm->top->nodenum_, batchsize_);
      
      //Initializes the update vectors and matrix to 0.
      WUpdate = gsl_matrix_float_calloc(rbm->top->nodenum_, rbm->bot->nodenum_);
      topBiasUpdate = gsl_vector_float_alloc(rbm->top->nodenum_);
      botBiasUpdate = gsl_vector_float_alloc(rbm->bot->nodenum_);
      identity = gsl_vector_float_alloc(batchsize_);
      gsl_vector_float_set_all(identity, 1);
      costterm = gsl_matrix_float_alloc(rbm->bot->weights_->size1, rbm->bot->weights_->size2);
      wsparsepenalty = gsl_matrix_float_alloc(rbm->top->nodenum_, rbm->bot->nodenum_);
      sparsepenalty = gsl_vector_float_alloc(rbm->top->nodenum_);
   }
   
   void teach(RBM*, Input_t *input);
};

class PCD : public Learner{
   
};

#endif
