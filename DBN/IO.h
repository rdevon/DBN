//
//  IO.h
//  DBN
//
//  Created by Devon Hjelm on 11/13/12.
//
//

#ifndef __DBN__IO__
#define __DBN__IO__

#include <iostream>
#include <vector>
#include <H5Cpp.h>

class MLP;
class DBN;
class Level;
class Layer;
class Connection;
class Layer_Monitor;

void save(const MLP&);
MLP load_MLP(std::string file_name);

void save(const Level&, const std::vector<Layer*>& layers, H5::Group& group);
Level load_level(H5::Group&, std::vector<Layer*>&);

void save(const std::vector<Connection*>&, const std::vector<Layer*>&, H5::Group&);
std::vector<Connection*> load_connections(H5::Group&, std::vector<Layer*>&);
void save(const Connection&, H5::DataSet &ds);
Connection *load_connection(H5::DataSet&, Layer*, Layer*);

void save(const std::vector<Layer*>&, H5::Group&);
std::vector<Layer*> load_layers(H5::Group&);
void save(const Layer&, H5::DataSet &ds);
Layer *load_layer(H5::DataSet&);

void save_features(Layer_Monitor*, std::string name);

herr_t file_info(hid_t loc_id, const char *name, void *list);

#endif /* defined(__DBN__IO__) */
