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
#include "MLP.h"
#include "Teacher.h"
#include "Viz.h"

class Layer;

/////////////////////////////////////
// Connection class
/////////////////////////////////////

// Binary connections between layers.  This is where the propagation happens as well as being a container for
// layers.  Layers are kept inside binary connections.  This should be make it easy to make generic graphs
// of layers.
class Connection : public LearningUnit, public Edge, public Monitor_Unit {
public:
   float freeEnergy;
   gsl_matrix_float *weights;
   
   gsl_vector_float *node_projections; //For getting projections onto to nodes
   
   Connection(Layer *from, Layer *to);
   
   void make_batch(int batchsize);
   
   void init_activation(MLP *mlp);
   int transmit_signal(Sample_flag_t s_flag);
   
   void catch_stats(Stat_flag_t, Sample_flag_t);
   void update(ContrastiveDivergence*);
   
   void init_data();
   int load_data(Data_flag_t, Sample_flag_t s_flag);
   
   void getFreeEnergy();
};

#endif /* defined(__DBN__Connections__) */