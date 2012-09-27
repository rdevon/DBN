//
//  Pathway.h
//  DBN
//
//  Created by Devon Hjelm on 9/27/12.
//
//

#ifndef __DBN__Pathway__
#define __DBN__Pathway__

#include <iostream>

#include "Types.h"
#include "MLP.h"

class Layer;
class Connection;

class Pathway : public MLP {
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
   
   Pathway(edge_list_t edges) {
      for (auto edge:edges) path.push_back((Connection*)edge);
   }
   
   void add_connection(Connection* connection);
   void make_batch(int batch_size);
   
   void set_direction_flag();
   
   void transmit_down();
   void transmit_up();
   void transmit();
};

#endif /* defined(__DBN__Pathway__) */
