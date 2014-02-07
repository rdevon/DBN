//
//  Autoencoder.cpp
//  DBN
//
//  Created by Devon Hjelm on 1/5/13.
//
//

#include "Autoencoder.h"
#include "RBM.h"
#include "Connections.h"
#include "Layers.h"
#include "Teacher.h"
#include <math.h>
#include "SupportFunctions.h"
#include "DataSets.h"

Autoencoder::Autoencoder(MLP &mlp) : MLP(mlp) {
   encoder = mlp;
   mlp.get_reconstruction_cost();
   
   Level level = mlp.levels.back();
   Level conj_level;
   
   std::map<Layer*, Layer*> l_newl;
   for (auto layer:level.top_layers) {
      
      l_newl[layer] = layer;
   }
   
   for (std::vector<Level>::reverse_iterator rliter = mlp.levels.rbegin(); rliter != mlp.levels.rend(); ++rliter) {
      level = *rliter;
      for (auto layer:level.bot_layers) {
         Layer *clone = layer->clone();
         l_newl[layer] = clone;
         if (x_is_in(layer, data_layers)) data_map[layer] = clone;
         else                             std::swap(layer->biases, layer->rec_biases);
      }
      for (auto connection:level.connections) {
         Connection *conj_conn = new Connection(*connection, l_newl[connection->to], l_newl[connection->from], true);
         conj_level.add(*conj_conn);
      }
      this->add(conj_level);
      decoder.add(conj_level);
      conj_level.clear();
   }
   
   for (auto layer:layers) if (!x_is_in(layer, data_layers)) layer->data = NULL;
   
   std::cout << *this;
}

void Autoencoder::learn(Gradient_Descent& teacher) {
   teacher.teachAE(*this);
}

void Gradient_Descent::teachAE(Autoencoder &ae) {
   momentum = 0.5;
   learning_multiplier = 1;
   int train_size = (int)ae.data_layers[0]->data->train.dim1;
   for (auto layer:ae.layers) {
      layer->set_defaults();
      layer->learning_rate/=500;
      layer->learning_rate/=train_size;
      layer->noisy = true;
   }
   for (auto layer:ae.data_layers) {
      layer->noisy = false;
      ae.data_map[layer]->noisy = false;
   }
   
   for (auto connection:ae.connections) {
      Layer *to = connection->to, *from = connection->from;
      to->set_with_fanin(*from);
      connection->learning_rate = to->learning_rate;
      connection->decay_type = NONE;
      //connection->apply_momentum = false;
      if (connection->from->data && connection->from->data->type == AOD_STIM) connection->transmission_scale = 10;
   }
   
   std::cout << "Fine tuning with autoencoder" << std::endl;
   float last_err = INFINITY;
   epochs = 100;
   int epoch = 0;
   while (epoch < epochs) {
      float err = 0;
      float ave_dim_err = 0;
      
      ae.init_data();
      ae.pull_data(TESTING);
      ae.transmit(UP, TESTING, NOSAMPLE);
      
      Level level = ae.levels.back();
      
      for (auto layer:ae.data_layers) {
         Layer *conj = ae.data_map[layer];
         err += distance(conj->m_testing, layer->m_testing);
         ave_dim_err = sqrtf(err) / layer->m_testing.dim2;
      }
      
      std::cout << "Epoch: " << epoch << " Squared error loss: " << err << std::endl;
     // std::cout << "         Average error per dim: " << ave_dim_err << std::endl;
     // std::cout << "         Class error: " << class_error << std::endl;
      
      ae.init_data();
      bool data_reset = false;
      do {
         //Init, pull data, and forward pass
         
         for (auto layer:ae.layers) layer->make_noise();
         data_reset = !((bool)ae.pull_data(LEARNING));
         ae.transmit(UP, LEARNING, NOSAMPLE);
         
         for (auto layer:ae.layers) layer->get_derivatives();
         
         //Gradient of data layers
         for (auto layer:ae.data_layers) {
            Layer *conj = ae.data_map[layer];
            conj->gradient *= (conj->m_learning - layer->m_learning);
         }
         
         //Backprop
         
         for (std::vector<Level>::reverse_iterator rliter = ae.levels.rbegin(); rliter != ae.levels.rend(); ++rliter) {
            level = *rliter;
            
            for (auto layer:level.bot_layers) {
               Matrix deltas(layer->gradient.dim1, layer->gradient.dim2);
               for (auto connection:level.connections) if (connection->from == layer)
                  deltas.times_plus(connection->to->gradient, connection->weights, 1, 1, CblasNoTrans, CblasNoTrans);
               layer->gradient *= deltas;
            }
            
            for (auto connection:level.connections)
               connection->gradient.times_plus(connection->to->gradient, connection->from->m_learning, 1, 0, CblasTrans, CblasNoTrans);
         }
         
         //Update the parameters
         
         for (auto connection:ae.connections) {
            connection->gradient *= -connection->learning_rate/connection->to->m_learning.dim1;
            connection->update(*this, false);
         }
         
         for (auto layer:ae.layers) {
            layer->gradient = layer->gradient.mean_image();
            layer->gradient *= -layer->learning_rate;
            layer->update(*this,false);
         }
      } while (!data_reset);
      ++epoch;
      if (learning_multiplier > 1) learning_multiplier *=.5;
      else learning_multiplier = 1;
#if 0
      if (err > last_err) break;
#endif
      last_err = err;
   }
}