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

class RBM : public Level {
public:
   
   ~RBM(){}
   RBM(Level);

   void reset();
   
   void gibbs_HV();
   void gibbs_VH();

   void init();
   
   void learn(ContrastiveDivergence&);
   
   void update(ContrastiveDivergence&, bool);
   void catch_stats(Stat_flag_t s);
   float get_reconstruction_cost();
};

#endif
