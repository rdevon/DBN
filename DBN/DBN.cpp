//
//  DBN.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/27/12.
//
//

#include "DBN.h"
#include "Connections.h"
#include "RBM.h"
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

void DBN::add_connection(Connection *connection) {
   edges.push_back(connection);
}

void DBN::finish_setup(){
   set_direction_flag_all(FORWARD);
   set_status_all(WAITING);
   
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter){
      Edge *edge = *e_iter;
      
      if (edge->from->input_edge != NULL) {
         edge->probe_network_structure(this, std::vector<Edge *>());
      }
   }
   set_status_all(WAITING);
   // Have to do it twice to get the levels right.
   
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter){
      Edge *edge = *e_iter;
      if (edge->from->input_edge != NULL) {
         std::vector<Edge *> path = edge->probe_network_structure(this, std::vector<Edge *>());
         Pathway *pathway = new Pathway;
         for (edge_list_iter_t e_iter = path.begin(); e_iter != path.end(); ++e_iter){
            Connection *connection  = (Connection*)(*e_iter);
            pathway->add_connection(connection);
         }
         pathways.push_back(pathway);
      }
   }
   /*for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Edge *edge = *e_iter;
      if (edge->level > rbms.size()) {
         RBM rbm;
         rbms.push_back(rbm);
      }
   }
   
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      
      Connection *connection = (Connection*)*e_iter;
      RBM *rbm = &rbms[connection->level-1];
      
      rbm->add_connection(connection);
      
   }*/
}

void DBN::learn(){
   
   set_status_all(WAITING);
   int layer = 1;
   while (1){
      set_status_all(DONE);
      RBM rbm;
      //rbm.viz = viz;
      rbm.teacher = teacher;
      
      for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter){
         Edge *edge = *e_iter;
         if (edge->level == layer)
            rbm.add_connection((Connection*)edge);
      }
      
      for (pathway_list_iterator_t p_iter = pathways.begin(); p_iter != pathways.end(); ++p_iter){
         Pathway *pathway = *p_iter;
         Connection *connection = pathway->path[layer-1];
         
         pathway->last = connection;
      }
      
      if (rbm.edges.size() == 0) break;
      
      for (int epoch = 1; epoch < 500; ++epoch){
         std::cout << "Layer " << layer << ", Epoch " << epoch << std::endl;
         rbm.learn();
      }
      rbm.transport_data();
      
      std::cout << "Done with " << layer << " layer." << std::endl;
      ++layer;
   }
}