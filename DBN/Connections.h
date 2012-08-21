//
//  Connections.h
//  DBN
//
//  Created by Devon Hjelm on 8/15/12.
//
//

#ifndef __DBN__Connections__
#define __DBN__Connections__

#include <iostream>
#include "Layers.h"

#endif /* defined(__DBN__Connections__) */

class Activator;

/////////////////////////////////////
// Connection class
/////////////////////////////////////

// Binary connections between layers.  This is where the propagation happens as well as being a container for
// layers.  Layers are kept inside binary connections.  This should be make it easy to make generic graphs
// of layers.
class Connection : public Learner {
public:
   Layer *top_;
   Layer *bot_;
   
   float freeEnergy;
   
   gsl_matrix_float *weights_;    //The weights between layers;
   
   Connection(Layer *bot, Layer *top);
   void update(ContrastiveDivergence*);
   void initializeWeights();
   void prop(Up_flag_t up, Sample_flag_t s);
   void initprop(Up_flag_t up);
   void getFreeEnergy();
   
   void makeBatch(int batchsize);
   void expandBiases();
   void catch_stats(Stat_flag_t s);
};

/////////////////////////////////////
// Activator class
/////////////////////////////////////

// The activator class.  This is for activating a set of layers given a set of connections.  This
// can activate all of the layers that are up or down to a set of connections.  This is nice for multimodal learning.
class Activator{
public:
   Connection *c1_, *c2_, *c3_;
   
   Up_flag_t up_flag_;
   Sample_flag_t s_flag_;
   
   ~Activator(){}
   Activator(Up_flag_t up, Connection *c1, Connection *c2 = NULL, Connection *c3 = NULL) : up_flag_(up), c1_(c1), c2_(c2), c3_(c3), s_flag_(SAMPLE) {}
   
   void activate();
};