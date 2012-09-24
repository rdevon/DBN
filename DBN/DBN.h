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

class Pathway : public MLP, public Monitor_Unit {
public:
   typedef std::vector<Connection*>                connection_list_t;
   typedef connection_list_t::iterator             connection_list_iterator_t;
   typedef connection_list_t::reverse_iterator     connection_list_reverse_iterator_t;
   
   connection_list_t                               path;
   
   Connection                                      *first, *last;
   
   Pathway(){}
   
   Pathway(Connection* base){
      path = std::vector<Connection*>();
      path.push_back(base);
   }
   
   void add_connection(Connection* connection);
   void make_batch(int batch_size);
   
   void transmit_down();
   void transmit_up();
};

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
