//
//  Teacher.h
//  DBN
//
//  Created by Devon Hjelm on 8/10/12.
//
//

#ifndef __DBN__Teacher__
#define __DBN__Teacher__

#include "Params.h"
#include "Matrix.h"

// This class keeps tracks of statistics and impliments teaching to various components.

class RBM;
class Monitor;
class Autoencoder;

class Teacher {
public:
   bool           learning;
   Monitor        *monitor;
   float          learning_multiplier;
   int            epochs;
   int            learning_count;
   int            k;
   
   float          momentum;
   
   ~Teacher(){}
   Teacher(){}
   Teacher(int epochs, float momentum = .5) : epochs(epochs), momentum(momentum), k(1), learning_multiplier(50), learning_count(0), learning(true), monitor(NULL) {}
   
   void multiply_rate();
   void divide_rate();
   void check_early_stop(RBM *rbm, int &epoch);
};

class Gradient_Descent : public Teacher {
public:
   Gradient_Descent(int momentum) : Teacher(0,momentum) {}
   void teachAE(Autoencoder &ae);
};

class ContrastiveDivergence : public Teacher {
public:
   
   ~ContrastiveDivergence(){}
   ContrastiveDivergence(){}
   ContrastiveDivergence(int epochs) : Teacher(epochs) {}
   
   void getStats(RBM&);
   void teachRBM(RBM &rbm);
};

class LearningUnit {
public:
   Decay_t                 decay_type;
   bool                    gain, apply_momentum;
   float                   learning_rate;
   float                   decay_rate;
   float                   weight_max_length;
   
   Matrix                  *param;
   
   Matrix                  learning_gain;
   
   Matrix                  gradient;
   Matrix                  acc_gradient;
   Matrix                  prev_gradient;
   
   Matrix                  velocity;
   Matrix                  acc_velocity;
   Matrix                  prev_velocity;
   
   LearningUnit(size_t dim1, size_t dim2):
   learning_gain(dim1,dim2), gradient(dim1,dim2), acc_gradient(dim1,dim2), prev_gradient(dim1,dim2),
   velocity(dim1,dim2), acc_velocity(dim1,dim2), prev_velocity(dim1,dim2),
   weight_max_length(0), decay_type(L2NORM), gain(false), apply_momentum(false) {}
   
   LearningUnit(const LearningUnit& other) :
   learning_rate(other.learning_rate), decay_rate(other.decay_rate), decay_type(other.decay_type), weight_max_length(other.weight_max_length),
   learning_gain(other.learning_gain), gradient(other.gradient), acc_gradient(other.acc_gradient), prev_gradient(other.prev_gradient), velocity(other.velocity),
   acc_velocity(other.acc_velocity), prev_velocity(other.prev_velocity), apply_momentum(false), gain(false){}
   
   virtual void update(Teacher&, bool apply_gain = false);
   virtual void catch_stats(Stat_flag_t)=0;
};

#endif /* defined(__DBN__Teacher__) */
