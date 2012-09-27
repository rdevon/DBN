//
//  RBM.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include "SupportFunctions.h"
#include "RBM.h"
#include "Connections.h"
#include "Layers.h"

RBM::RBM () {free_energy = 0;}

void RBM::add_connection(Connection *connection){
   edges.push_back(connection);
}

void RBM::getFreeEnergy(){
}

void RBM::learn(){
   teacher->teachRBM(this);
}

void RBM::prop(Edge_direction_flag_t dir){
   
   set_direction_flag_all(dir);
   set_status_all_endpoints(WAITING);
   
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end() ; ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      connection->init_activation(this);
   }
   
   int to_transmit = (int)edges.size();
   edge_list_iter_t e_iter = edges.begin();
   while (to_transmit > 0) {
      Connection *connection = (Connection*)(*e_iter);
      if (connection->transmit_signal(sample_flag))
         to_transmit-=1;
      ++e_iter;
      if (e_iter == edges.end()) e_iter = edges.begin();
   }
}

void RBM::gibbs_HV(){
   prop(FORWARD);
   prop(BACKWARD);
}

void RBM::gibbs_VH(){
   prop(BACKWARD);
   prop(FORWARD);
}

void RBM::update(ContrastiveDivergence *cd){
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      connection->update(cd);
   }
}

void RBM::catch_stats(Stat_flag_t stat_flag){
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      connection->catch_stats(stat_flag, SAMPLE);
   }
}

void RBM::getReconstructionCost(){
   init_data();
   
   load_data(TRAIN);
   
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      connection->make_batch(0); //A hack to pass down that we want the entire input
      Layer *from_layer = (Layer*)connection->from;
      gsl_matrix_float_memcpy(from_layer->extra, from_layer->samples);
   }
   
   sample_flag = NOSAMPLE;
   
   gibbs_HV();
   
   reconstruction_cost = 0;
   
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      Layer *from_layer = (Layer*)connection->from;
      reconstruction_cost += from_layer->reconstructionCost(from_layer->extra, from_layer->samples);
      
      std::cout << "Reconstruction cost: " << from_layer->reconstruction_cost << std::endl;
   }
}


void RBM::transport_data(){
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      connection->make_batch(0); //A hack to pass down that we want the entire input
      Layer *from_layer = (Layer*)connection->from;
      gsl_matrix_float_transpose_memcpy(from_layer->samples, from_layer->input_edge->train);
   }
   sample_flag = NOSAMPLE;
   prop(FORWARD);
   
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      connection->make_batch(0); //A hack to pass down that we want the entire input
      Layer *to_layer = (Layer*)connection->to;
      InputEdge *input_edge = new InputEdge;
      input_edge->train = gsl_matrix_float_alloc(to_layer->samples->size2, to_layer->samples->size1);
      gsl_matrix_float_transpose_memcpy(input_edge->train, to_layer->samples);
      to_layer->input_edge = input_edge;
   }
}

void RBM::make_batch(int batch_size){
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      connection->make_batch(batch_size);
   }
}

int RBM::load_data(Data_flag_t d_flag){
   set_status_all(WAITING);
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      Layer *from_layer= (Layer*)connection->from;
      if (!from_layer->load_data(d_flag, sample_flag)) return 0;
   }
   return 1;
}

void RBM::init_data(){
   
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      Connection *connection = (Connection*)(*e_iter);
      connection->init_data();
   }
   
}