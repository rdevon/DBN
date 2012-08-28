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

class Learner;
class Visualizer;
class RBM;

class Teacher{
public:
   ~Teacher(){}
   Teacher(){}
   virtual void teachRBM(RBM* rbm) = 0;
};

class ContrastiveDivergence : public Teacher {
public:
   float momentum_, sparsitycost_, p_, lambda_;
   int k_, batchsize_;
   gsl_vector_float *identity, *forvizvec;
   
   Visualizer *viz_;
   
   ~ContrastiveDivergence(){}
   ContrastiveDivergence(){}
   ContrastiveDivergence(RBM *rbm, float momentum, int k, float p, float lambda, float sparsitycost, int batchsize);
   
   void getStats(RBM*);
   void teachRBM(RBM* rbm);
   void monitor(RBM*, int i);
};

class Learner{
public:
   Learner(){}
   Teacher *teacher;
   virtual void learn() = 0;
};

class LearningUnit {
public:
   float learning_rate_;
   float decay_;
   gsl_vector_float *vec_update;
   gsl_vector_float *vec_update2;
   gsl_matrix_float *mat_update;
   virtual void update(ContrastiveDivergence*) = 0;
   gsl_matrix_float *stat1, *stat2, *stat3, *stat4;
   LearningUnit(){decay_ = 0;}
};

#endif /* defined(__DBN__Teacher__) */
