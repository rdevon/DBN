//
//  MLP.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/30/12.
//
//

#include "MLP.h"
#include "Connections.h"
#include "DataSets.h"
#include "SupportMath.h"
#include "SupportFunctions.h"
#include "Layers.h"

//------ Level Methods

void Level::add(Connection& connection) {
   if (!check_connection(connection)) error("Possible loop or bad level");
   connections.push_back(&connection);
   if (!x_is_in(connection.from, bot_layers)) bot_layers.push_back(connection.from);
   if (!x_is_in(connection.to, top_layers))   top_layers.push_back(connection.to);
}

bool Level::check_connection (Connection &connection) {
   for (auto layer:top_layers) if (layer == connection.from) return false;
   for (auto layer:bot_layers) if (layer == connection.to) return false;
   if (connection.to == connection.from) return false;
   return true;
}

Transmit_t Level::transmit(Direction_t direction, Purpose_t purpose, Sample_flag_t sample) {
   switch (direction) {
      case UP:   for (auto layer:top_layers) layer->reset_switch = 0; break;
      case DOWN: for (auto layer:bot_layers) layer->reset_switch = 0; break;
   }
   for (auto connection:connections) if (connection->transmit_signal(direction, purpose, sample) == TRANSMIT_FAILURE) return TRANSMIT_FAILURE;
   switch (direction) {
      case UP:   for (auto layer:top_layers) layer->finish_activation(purpose, sample); break;
      case DOWN: for (auto layer:bot_layers) layer->finish_activation(purpose, sample); break;
   }
   return TRANSMIT_SUCCESS;
}

void Level::init_data() {
   for (auto layer:bot_layers) {
      if (!layer->data) error("Level must have data at every bottom layer to init");
      layer->data->index = 0;
   }
}

int Level::pull_data(Purpose_t purpose) {
   int data_switch = 1;
   for (auto layer:bot_layers) {if (!layer->pull_data(purpose)) data_switch = 0;}
   return data_switch;
}

void Level::transport_data(Level &to_level) {
   std::cout << "Transporting data from layer" << std::endl;
   std::cout << *this;
   std::cout << "to" << std::endl;
   std::cout << to_level;
   int datasize = -1;
   for (auto layer:bot_layers) {
      if (!layer->data) error("All bottom layers must have data to transport");
      if (layer->data->dims[3] != datasize && datasize != -1) error("Transporting data of different sizes");
      else datasize = (int)layer->data->dims[3];
   }
   
   for (auto layer:top_layers) layer->data = new DataSet(datasize, layer->nodenum);
   for (auto connection:connections) connection->transmit_data(connection->from->data->data, connection->to->data->data);
   
   for (auto layer:top_layers) {
      Matrix &out = layer->data->data;
      out += layer->biases;
      switch (layer->type) {
         case RELU:    out.apply([](float &x) {x = softplus(x);}); break;
         case SOFTMAX: out.apply([](float &x) {x = expf(x);}); out.norm_rows(); break;
         case SIGMOID: out.apply([](float &x) {x = sigmoid(x);}); break;
         default: break;
      }
      layer->data->make_validation();
   }
}

void Level::clear() {top_layers.clear(); bot_layers.clear(); connections.clear();}

//------ MLP Methods

void MLP::add(Level& level) {
   for (auto connection:level.connections) if (!x_is_in(connection, connections)) connections.push_back(connection);
   for (auto layer:level.bot_layers) {
      if ((!layer->data) && levels.size() != 0 && !x_is_in(layer, levels.back().top_layers)) error("When connecting levels, non-data layers must correspond to top level layers");
      if (!x_is_in(layer, layers)) layers.push_back(layer);
   }
   for (auto layer:level.top_layers) if (!x_is_in(layer, layers)) layers.push_back(layer);
   levels.push_back(level);
}

void MLP::init_data() {for (auto layer:data_layers) layer->data->index = 0;}

int MLP::pull_data(Purpose_t purpose) {
   int data_switch = 1;
   for (auto layer:data_layers) if(!layer->pull_data(purpose)) data_switch = 0;
   return data_switch;
}

void MLP::transmit(Direction_t direction, Purpose_t purpose, Sample_flag_t sample) {
   switch (direction) {
      case UP: {
         int l = 0;
         for (auto level:levels) {
            if (purpose == RECOGNITION && l == levels.size()-1) purpose = TESTING;
            if (level.transmit(direction, purpose, sample) == TRANSMIT_FAILURE) error("Transmit failure");
            ++l;
         }
      } break;
      case DOWN: {
         for (auto r_iter = levels.rbegin(); r_iter != levels.rend(); ++r_iter)
            if ((*r_iter).transmit(direction, purpose, sample) == TRANSMIT_FAILURE) error("Transmit failure");
      } break;
   }
}

Transmit_t MLP::transmit(Layer*, Layer*) {error("layer to layer transmission not online"); return TRANSMIT_FAILURE;}

float MLP::get_reconstruction_cost(){
   
   reconstruction_cost = 0;
   
   std::vector<Matrix> data_mats;
   
   init_data();
   pull_data(TESTING);
   transmit(UP, RECOGNITION, NOSAMPLE);
   for (auto layer:data_layers) data_mats.push_back(layer->m_testing);
   transmit(DOWN, GENERATING, NOSAMPLE);
   
   auto d_iter = data_mats.begin();
   for (auto layer:data_layers) {
      reconstruction_cost += layer->reconstructionCost(*d_iter, layer->m_testing);
      ++d_iter;
   }
   
   std::cout << "MLP reconstruction costs: " << reconstruction_cost;
   
   int l = 0;
   for (auto level:levels) {
      level.init_data();
      level.pull_data(TESTING);
      Purpose_t purpose;
      if (l == levels.size()-1) purpose = TESTING;
      else purpose = RECOGNITION;
      level.transmit(UP, purpose, NOSAMPLE);
      data_mats.clear();
      for (auto connection:level.connections) data_mats.push_back(connection->from->m_testing);
      level.transmit(DOWN, TESTING, NOSAMPLE);
      float level_reconstruction_cost = 0;
      
      d_iter = data_mats.begin();
      for (auto connection:level.connections) {
         level_reconstruction_cost += connection->from->reconstructionCost(*d_iter, connection->from->m_testing);
         ++d_iter;
      }
      
      std::cout << "\t| l = " << convert_to_string(l) << ": " << level_reconstruction_cost;
      ++l; 
   }
   std::cout << std::endl;
   
   return reconstruction_cost;
}

//------ OSTREAM stuff

std::ostream& operator<<(std::ostream& out,MLP& mlp) {
   int l = 1;
   for (auto level:mlp.levels) {
      out << "Level " << convert_to_string(l) << std::endl << level;
      ++l;
   }
   out << "Data Layers:";
   for (auto layer:mlp.data_layers) out << "     " << *layer << std::endl;
   return out;
}

std::ostream& operator<<(std::ostream& out,Level &level) {
   for (auto connection:level.connections) out << *connection << std::endl;
   out << "------------" << std::endl;
   return out;
}
std::ostream& operator<<(std::ostream& out,Connection &connection) {
   out << "  " << *(connection.from) << " -> " << *(connection.to);
   return out;
}

std::ostream& operator<<(std::ostream& out, Layer &layer) {
   out << layer.type << " Layer [" << &layer << "] (" << layer.nodenum << "x" << layer.m_learning.dim1 << "x" << layer.m_testing.dim1 << ")";
   if (layer.data) out << "(D)";
   return out;
}

