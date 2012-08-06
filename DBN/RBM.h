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

class RBM;

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
   gsl_vector_float *oldnormterm_;
   gsl_vector_float *newnormterm_;
   
   Visualizer *viz;
   
   ~CD(){}
   CD(){}
   CD(float learningRate, float weightcost, int k, float p, float lambda, float sparsitycost, Visualizer *v, int batchsize=1) : learningRate_(learningRate), weightcost_(weightcost), k_(k), p_(p), lambda_(lambda), sparsitycost_(sparsitycost), viz(v), batchsize_(batchsize) {}
   
   void teach(RBM*, Input_t *input);
};

class PCD : public Learner{

};

class RBM {
public:
   
   float freeEnergy_, reconstructionError_;
   
   Layer *top;
   Layer *bot;
   
   Activator activator;
   
   ~RBM(){}
   RBM(){}
   RBM(Layer*);
   RBM(Layer*,Layer*);
   
   void getFreeEnergy();
   void gibbs_HV();
   void gibbs_VH();
   void getReconstructionError(Input_t *input);
   void sample(DataSet *data, Visualizer *viz);
};


#endif
