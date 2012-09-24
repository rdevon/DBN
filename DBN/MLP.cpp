//
//  MLP.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/30/12.
//
//

#include "MLP.h"
#include "IO.h"
#include "SupportMath.h"
#include "SupportFunctions.h"

void MLP::set_direction_flag_all(Edge_direction_flag_t dir) {
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) (*e_iter)->direction_flag = dir;
}

void MLP::set_status_all_endpoints(Node_status_flag_t stat) {
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter)
      if       ((*e_iter)->direction_flag == FORWARD)     (*e_iter)->to->status = stat;
      else if  ((*e_iter)->direction_flag == BACKWARD)    (*e_iter)->from->status = stat;
}

void MLP::set_status_all(Node_status_flag_t stat) {
   for (edge_list_iter_t e_iter = edges.begin(); e_iter != edges.end(); ++e_iter) {
      (*e_iter)->to->status = stat;
      (*e_iter)->from->status = stat;
   }
}

InputEdge::InputEdge(DataSet *data) : train(data->train), test(data->test), input_x(data->width), input_y(data->height), name(data->name), mask(data->mask){std::cout << "Forming input edge" << std::endl;}

void InputEdge::apply_mask(gsl_vector_float *sample, gsl_vector_float *sample_with_mask){
   if (mask == NULL) {
      gsl_vector_float_memcpy(sample_with_mask, sample);
      
      return;
   }
   
   for (int i = 0, iprime = 0; i < mask->size; ++i) {
      float maskval = gsl_vector_float_get(mask, i);
      if (maskval == 1) {
         float val = gsl_vector_float_get(sample, iprime);
         gsl_vector_float_set(sample_with_mask, i, val);
         ++iprime;
      }
      else gsl_vector_float_set(sample_with_mask, i, WHITE);
   }
}

std::vector<Edge*> Edge::probe_network_structure(MLP *mlp, std::vector<Edge*> pathway){
   
   if (to->status == DONE) return pathway;
   pathway.push_back(this);
   
   std::vector<Edge*> valid_edges;
   
   for (std::vector<Edge*>::iterator e_iter = to->forward_edges.begin(); e_iter != to->forward_edges.end(); ++e_iter){
      Edge *edge = *e_iter;
      if (std::find(mlp->edges.begin(), mlp->edges.end(), edge) != mlp->edges.end()) valid_edges.push_back(edge);
   }
   
   for (std::vector<Edge*>::iterator e_iter = valid_edges.begin(); e_iter != valid_edges.end(); ++e_iter){
      Edge *edge = *e_iter;
      pathway = edge->probe_network_structure(mlp, pathway);
   }
   
   if (valid_edges.size() == 0) {
      if (level < pathway.size()) level = (int)pathway.size();
   }
   
   else {
      level = 0;
      for (std::vector<Edge*>::iterator e_iter = valid_edges.begin(); e_iter != valid_edges.end(); ++e_iter){
         Edge *edge = *e_iter;
         if (edge->level > level-1) level = edge->level - 1;
      }
   }
   
   to->status = DONE;
   return pathway;
}