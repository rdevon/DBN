//
//  Teacher.h
//  DBN
//
//  Created by Devon Hjelm on 8/10/12.
//
//

#ifndef __DBN__Teacher__
#define __DBN__Teacher__

#include "Types.h"

// This class keeps tracks of statistics and impliments teaching to various components.

class RBM;
class Learning_Monitor;

class Teacher{
public:
   Learning_Monitor        *monitor;
   
   ~Teacher(){}
   Teacher(){}
   virtual void teachRBM(RBM* rbm) = 0;
};

class ContrastiveDivergence : public Teacher {
public:
   float                   momentum;
   int                     k,
                           batchsize,
                           epochs;
                           
   
   gsl_vector_float        *identity;
   
   ~ContrastiveDivergence(){}
   ContrastiveDivergence(){}
   ContrastiveDivergence(float momentum, int k, int batchsize, int epochs);
   
   void getStats(RBM*);
   void teachRBM(RBM* rbm);
};

class Learner{
public:
   Teacher                 *teacher;
   
   Learner(){}
   virtual void learn() = 0;
};

class LearningUnit {
public:
   bool                    learning_up_to_date;
   float                   learning_rate;
   float                   decay;
   gsl_vector_float        *vec_update;
   gsl_vector_float        *vec_update2;
   gsl_matrix_float        *mat_update;
   gsl_matrix_float        *stat1, *stat2, *stat3, *stat4;
   
   LearningUnit(){}
   
   virtual void update(ContrastiveDivergence*)=0;
   virtual void catch_stats(Stat_flag_t, Sample_flag_t)=0;
};

#endif /* defined(__DBN__Teacher__) */
