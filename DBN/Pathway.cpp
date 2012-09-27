//
//  Pathway.cpp
//  DBN
//
//  Created by Devon Hjelm on 9/27/12.
//
//

#include "Pathway.h"
#include "Connections.h"
#include "Layers.h"


void Pathway::add_connection(Connection *connection){
   path.push_back(connection);
   edges.push_back(connection);
   if (path.size() == 1)
      first = path[0];
}

void Pathway::make_batch(int batch_size){
   Node *bottom = first->from;
   if (batch_size==0) batch_size = (int)bottom->input_edge->train->size1;
   
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      connection->make_batch(batch_size);
   }
}

void Pathway::transmit_down(){
   set_direction_flag_all(BACKWARD);
   
   connection_list_reverse_iterator_t c_iter = path.rbegin();
   
   for (; *c_iter != last; ++c_iter);
   
   for (; c_iter != path.rend(); ++c_iter) {
      Connection *connection = *c_iter;
      
      connection->transmit_signal(NOSAMPLE);
      Layer *endpoint = (Layer*)connection->from;
      endpoint->status = READY;
      endpoint->finish_activation(NOSAMPLE);
   }
}

void Pathway::transmit_up(){
   set_direction_flag_all(FORWARD);
   for (connection_list_iterator_t c_iter = path.begin(); c_iter != path.end(); ++c_iter) {
      Connection *connection = *c_iter;
      if (connection == last) break;
      connection->transmit_signal(NOSAMPLE);
      
      Layer *endpoint = (Layer*)connection->to;
      endpoint->status = READY;
      endpoint->finish_activation(NOSAMPLE);
   }
}

void Pathway::set_direction_flag(){
   Node *from_node = path[0]->from;
   for (auto connection:path) {
      if (connection->from == from_node) connection->direction_flag = FORWARD;
      else  connection->direction_flag = BACKWARD;
   }
}

void Pathway::transmit(){
   set_direction_flag();
   path[0]->from->status = DONE;
   set_status_all_endpoints(WAITING);
   Layer *endpoint;
   for (auto connection:path) {
      if (connection->direction_flag == FORWARD) endpoint = (Layer*)connection->to;
      else endpoint = (Layer*)connection->from;
      
   }
   
}