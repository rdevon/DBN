//
//  Get_Network.h
//  DBN
//
//  Created by Devon Hjelm on 11/14/12.
//
//

#ifndef __DBN__Get_Network__
#define __DBN__Get_Network__

#include <iostream>
#include "Params.h"

class DBN;
class MLP;
class Level;
class Value_Handler;
class Layer;

DBN *return_network();

DBN *return_aod_network(std::string filename, std::ofstream &log_stream, std::streambuf *psbuf);

DBN *load_and_stack(std::string filename, std::ofstream &log_stream, std::streambuf *psbuf);

void add_level(MLP &mlp, Level &prev_level, Value_Handler handler, Layer *data_layer, int l, int batchsize, int valid_size, gsl_rng *rand, bool ask_stim = true);

#endif /* defined(__DBN__Get_Network__) */
