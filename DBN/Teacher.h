//
//  Teacher.h
//  DBN
//
//  Created by Devon Hjelm on 8/10/12.
//
//

#ifndef __DBN__Teacher__
#define __DBN__Teacher__

#include <iostream>
#include "IO.h"

// This class keeps tracks of statistics and impliments teaching to various components.

class RBM;
class Visualizer;

class Teacher{
public:
   ~Teacher(){}
   Teacher(){}
};

class ContrastiveDivergence : public Teacher {
public:
   float learningRate_, weightcost_, momentum_, sparsitycost_, p_, lambda_;
   int k_, batchsize_;
   gsl_vector_float *identity, *forvizvec;
   
   Visualizer *viz_;
   RBM *rbm_;
   
   ~ContrastiveDivergence(){}
   ContrastiveDivergence(){}
   ContrastiveDivergence(RBM *rbm, float learningRate, float weightcost, float momentum, int k, float p, float lambda, float sparsitycost, int batchsize);
   
   void getStats();
   void run();
   void monitor(int i);
};

class Learner{
public:
   float learning_rate_;
   gsl_vector_float *vec_update;
   gsl_vector_float *vec_update2;
   gsl_matrix_float *mat_update;
   virtual void update(ContrastiveDivergence*) = 0;
   gsl_matrix_float *stat1, *stat2, *stat3, *stat4;
};





#endif /* defined(__DBN__Teacher__) */
