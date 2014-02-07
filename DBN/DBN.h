//
//  DBN.h
//  DBN
//
//  Created by Devon Hjelm on 8/27/12.
//
//

#ifndef __DBN__DBN__
#define __DBN__DBN__

#include <iostream>
#include "Teacher.h"
#include "MLP.h"
#include "Viz.h"

class DBN : public MLP {
public:
   bool monitor_dbn;
   MLP  reference_MLP;
   
   DBN(MLP reference_MLP);
   
   void init_learners(){}
   void learn(ContrastiveDivergence&);
   void view();
   void stack(Level &level);
};


#endif /* defined(__DBN__DBN__) */
