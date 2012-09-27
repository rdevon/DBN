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

class RBM;
class Connection;
class Pathway;

class DBN : public Learner, public MLP {
public:
   typedef std::vector<Pathway*>                    pathway_list_t;
   typedef pathway_list_t::iterator                pathway_list_iterator_t;
   
   pathway_list_t                                  pathways;
   
   DBN(){}
   DBN(edge_list_t e){
      edges = e;
   }
   
   void add_connection(Connection* connection);
   void finish_setup();
   void learn();
};

#endif /* defined(__DBN__DBN__) */
