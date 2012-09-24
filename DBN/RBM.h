//
//  RBM.h
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_RBM_h
#define DBN_RBM_h
#include "MLP.h"
#include "Teacher.h"

class Connection;

class RBM : public Learner, public MLP {
public:
   
   Sample_flag_t                       sample_flag;
   
   float                               free_energy;
   float                               reconstruction_cost;
   
   ~RBM(){}
   RBM();
   
   void add_connection(Connection* connection);
   
   void getFreeEnergy();
   void prop(Edge_direction_flag_t);
   void gibbs_HV();
   void gibbs_VH();
   
   void getReconstructionCost();
   
   void learn();
   
   void get_dims(float *topdim, float *botdim);
   void make_batch(int batch_size);
   
   void init_data();
   int load_data(Data_flag_t d_flag);
   void transport_data();
   
   void update(ContrastiveDivergence*);
   void catch_stats(Stat_flag_t s);
};

#endif
