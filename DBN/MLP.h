//
//  MLP.h
//  DBN
//
//  Created by Devon Hjelm on 8/30/12.
//
//

#ifndef __DBN__MLP__
#define __DBN__MLP__

#include "Params.h"

class Layer;
class Connection;

class Level {
   void swap(Level &other) {
      using std::swap;
      swap(connections, other.connections);
      swap(top_layers, other.top_layers);
      swap(bot_layers, other.bot_layers);
   }
   bool check_connection (Connection &connection);
   
public:
   std::vector<Connection*> connections;
   std::vector<Layer*> top_layers, bot_layers;
   Level(){}
   Level(const Level &other): connections(other.connections), top_layers(other.top_layers), bot_layers(other.bot_layers){}
   Level &operator= (Level rhs) {swap(rhs); return *this;}
   
   void add(Connection& connection);
   Transmit_t transmit(Direction_t direction, Purpose_t purpose, Sample_flag_t sample);
   void init_data();
   int pull_data(Purpose_t);
   void transport_data(Level &);
   void clear();
};

class MLP {
   
   void swap(MLP &other) {
      using std::swap;
      swap(levels, other.levels);
      swap(data_layers, other.data_layers);
      swap(layers, other.layers);
      swap(connections, other.connections);
      viz_layer = other.viz_layer;
   }

public:
   std::string name;
   
   float reconstruction_cost;
   Layer                      *viz_layer;
   std::vector<Layer*>        data_layers;
   std::vector<Level>         levels;
   std::map<std::pair<Layer*, Layer*>, Connection*>   graph;
   std::vector<Layer*>        layers;
   std::vector<Connection*>   connections;
   
   MLP(){}
   MLP &operator= (MLP rhs) {swap(rhs);return *this;}
   MLP(const MLP &other) : levels(other.levels), connections(other.connections), layers(other.layers), viz_layer(other.viz_layer), data_layers(other.data_layers) {}
   
   void add(Level& level);
   void transmit(Direction_t direction, Purpose_t purpose, Sample_flag_t sample);
   Transmit_t transmit(Layer*, Layer*);
   void init_data();
   int pull_data(Purpose_t purpose);
   float get_reconstruction_cost();
   
   //todo
#if 0
   void finish() {
      layers.clear();
      connections.clear();
      graph.clear();
      for (auto level:levels) {
         for (auto connection:level.connections) {
            graph[std::pair<Layer*, Layer*>(connection->from, connection->to)] = connection;
            if (!x_is_in(connection->from, layers)) layers.push_back(connection->from);
            if (!x_is_in(connection->to, layers)) layers.push_back(connection->to);
            if (!x_is_in(connection, connections)) connections.push_back(connection);
         }
      }
   }
   
   Connection &attach(Layer* l1, Layer* l2) {
      std::pair<Layer*, Layer*> ll(l1,l2);
      if (std::find(graph.begin(), graph.end(), ll) != graph.end()) return *graph[ll];
      else {
      }
   }
#endif
};

std::ostream& operator<<(std::ostream& out,MLP& mlp);
std::ostream& operator<<(std::ostream& out,Level &level);
std::ostream& operator<<(std::ostream& out,Connection &connection);
std::ostream& operator<<(std::ostream& out,Layer &layer);

#endif /* defined(__DBN__MLP__) */
