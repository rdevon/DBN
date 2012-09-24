//
//  MLP.h
//  DBN
//
//  Created by Devon Hjelm on 8/30/12.
//
//

#ifndef __DBN__MLP__
#define __DBN__MLP__

#include "Types.h"

class DataSet;
class Node;
class MLP;

class Edge {
public:
   
   Edge_direction_flag_t            direction_flag;
   
   Node                             *from, *to;
   
   int level;
   
   Edge(){}
   
   // METHODS ------------------------------------------------------------------------------------
   std::vector<Edge*> probe_network_structure(MLP*, std::vector<Edge*>);
   
};

class InputEdge {
public:
   Input_t *train;
   Input_t *test;
   int input_x, input_y;
   int index;
   gsl_vector_float *mask;
   std::string name;
   
   InputEdge(){};
   ~InputEdge(){}
   
   InputEdge(DataSet *data);
   void apply_mask(gsl_vector_float *sample, gsl_vector_float *sample_with_mask);
};

class Node{
public:
   
   typedef std::vector<Edge*>       edge_list_t;
   typedef edge_list_t::iterator    edge_list_iter_t;
   
   // MEMBERS -------------------------------------------------------------------------------------
   
   Node_status_flag_t               status;
   
   edge_list_t                      forward_edges;
   edge_list_t                      backward_edges;
   
   InputEdge                        *input_edge;
   
   int visits_waiting;
   
   // CONSTRUCTORS -------------------------------------------------------------------------------------
   
   Node() : status(DONE) {input_edge = NULL;}
};

class MLP{
public:
   typedef std::vector<Edge*>        edge_list_t;
   typedef edge_list_t::iterator     edge_list_iter_t;

   edge_list_t                       edges;
   int test;
  
   MLP(){test = 10;}
   
   void set_direction_flag_all(Edge_direction_flag_t);
   void set_status_all_endpoints(Node_status_flag_t);
   void set_status_all(Node_status_flag_t);
   void set_all_outside_off();
};

#endif /* defined(__DBN__MLP__) */
