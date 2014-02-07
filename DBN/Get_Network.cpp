
//
//  Get_Network.cpp
//  DBN
//
//  Created by Devon Hjelm on 11/14/12.
//
//

#include "Get_Network.h"
#include "Teacher.h"
#include "SupportFunctions.h"
#include "Connections.h"
#include "Layers.h"
#include "DataSets.h"
#include "DBN.h"
#include <string>
#include "IO.h"

struct Value_Handler {
   
   Dataset_t dataset;
   bool multimodal;
   
   int hidden_layer_number;
   Layer_t hidden_layer_type;
   int nodenum;
   
   float learning_rate;
   float decay;
   
   bool dropout;
   float bottleneck;
   float momentum;
   int k;
   int batchsize;
   int epochs;
   
   
   void print() {
      std::cout << "DBN parameters: " << std::endl;
      std::cout << "     Multimodal:           ";
      if (multimodal) std::cout << "yes"; else std::cout << "no";
      std::cout << std::endl;
      std::cout << "     Number Hidden Layers: " << hidden_layer_number << std::endl;
      std::cout << "     Hidden Layer Type:    " << hidden_layer_type << std::endl;
      std::cout << "     Hidden Nodes:         " << convert_to_string(nodenum) << std::endl;
      std::cout << "     Dropout               ";
      std::cout << "     Batchsize             " << convert_to_string(batchsize) << std::endl;
      if (dropout) std::cout << "yes"; else std::cout << "no";
      std::cout << std::endl;
      
   }
   
   bool visualization;
   
   Value_Handler(){
      dataset = SSL_VIS;
      visualization = true;
      dropout = false;
      multimodal = false;
      
      bottleneck = 1;
      momentum = 0;
      k = 1;
      batchsize = 1;
   }
   
   void load_MNIST_defaults() {
      dataset = MNIST;
      multimodal = false;
      hidden_layer_number = 1;
      hidden_layer_type = SIGMOID;
      nodenum = 200;
      batchsize = 15;
      epochs = 100;
      momentum = 0.5;
   }
   
   void load_SSL_VIS_defaults() {
      dataset = SSL_VIS;
      batchsize = 4;
      multimodal = false;
      hidden_layer_number = 1;
      hidden_layer_type = RELU;
      nodenum = 20;
      epochs = 1000;
      momentum = 0.5;
   }
   
   void load_VOL_VIS_defaults() {
      dataset = VOL_VIS;
      batchsize = 5;
      multimodal = false;
      hidden_layer_number = 1;
      hidden_layer_type = RELU;
      nodenum = 50;
      epochs = 30;
      momentum = 0.5;
   }
   
   void load_AOD_defaults() {
      dataset = AOD;
      batchsize = 5;
      multimodal = false;
      hidden_layer_number = 1;
      hidden_layer_type = RELU;
      nodenum = 50;
      epochs = 30;
      momentum = 0.5;
   }
   
   
   void load_SIGMOID_defaults() {learning_rate = AUTO, decay = AUTO;}
   void load_RELU_defaults() {learning_rate = AUTO, decay = AUTO;}
   void load_GAUSSIAN_defaults() {learning_rate = AUTO, decay = AUTO;}
   void load_SOFTMAX_defaults() {learning_rate = AUTO, decay = AUTO;}
   
   void print_dataset_Q() {
      std::cout << "Which dataset:" <<std::endl;
      std::cout << "                          1) MNIST" << std::endl;
      std::cout << "                          2) Single slice visuomotor (fMRI)" << std::endl;
      std::cout << "                          3) Volume visuomotor (fMRI)" << std::endl;
      std::cout << "                          4) AOD single subject (fMRI)" << std::endl;
   }
   void print_stim_query() {std::cout << "Do you want to include stimulus on this layer? (yes/no)";}
   void print_subject_query(int i) {std::cout << "Subject " << i+1 << "? (yes/no)" << std::endl;}
   void print_number_layers_Q() {std::cout << "How many hidden layers? (sorry multimodal not supported in this command-line)";}
   void print_layer_type_Q(int i) {
      std::cout << "For hidden layer " << i+1 << ", enter the type (WARNING: performance may vary for different datasets)" << std::endl;
      std::cout << "                                        1) Sigmoid" << std::endl;
      std::cout << "                                        2) Rectified Linear" << std::endl;
      std::cout << "                                        3) Gaussian" << std::endl;
      std::cout << "                                        4) Softmax" << std::endl;
   }
   void print_node_number_Q() {std::cout << "How many nodes?";}
   void print_learning_rate_Q() {std::cout << "Enter the learning rate for connection (may drastically change performance)";}
   void print_decay_rate_Q() {std::cout << "Enter the L2 decay for connection (may drastically change performance)";}
   void print_dropout_Q() {std::cout << "Dropout? (experimental) (yes/no)";}
   void print_momentum_Q() {std::cout << "Momentum";}
   void print_k_Q() {std::cout << "k (contrastive divergence gibbs cycles)";}
   void print_batchsize_Q() {std::cout << "Batch size";}
   void print_epochs_Q() {std::cout << "Epochs per layer (you can do early stopping";}
   void print_visualization_Q() {std::cout << "Visualization? (yes/no)(slows performance)";}
   
   
   void print_default(bool dval){
      std::string yesno;
      if (dval) yesno = "yes";
      else yesno = "no";
      std::cout << ": (" << yesno << ") ";
   }
   
   void print_default(Dataset_t dval) {
      std::string default_value;
      switch(dval) {
         case MNIST: default_value = "MNIST"; break;
         case SSL_VIS: default_value = "Single Slice fMRI"; break;
         case VOL_VIS: default_value = "3D fMRI"; break;
         case AOD: default_value = "AOD"; break;
         default: break;
      }
      std::cout << ": (" << default_value << ") ";
   }
   
   void print_default(Layer_t dval) {
      std::string default_value;
      switch(dval) {
         case SIGMOID: default_value = "Sigmoid"; break;
         case RELU: default_value = "Reclified Linear"; break;
         case GAUSSIAN: default_value = "Gaussian"; break;
         case SOFTMAX: default_value = "Softmax"; break;
      }
      std::cout << ": (" << default_value << ") ";
   }
   
   template <typename T> void print_default(T dval) {std::cout << ": (" << dval << ") ";}
   
   void get_value(bool *value) {
      std::cin.clear();
      std::string input;
      while (1) {
         std::string input_buffer;
         getline(std::cin, input_buffer);
         if (input_buffer == "") {
            return;
         }
         if (input_buffer == "yes" || input_buffer == "no") {
            input = input_buffer;
            break;
         }
         std::cout << "Invalid input" << std::endl;
         std::cin.clear();
      }
      if (input == "yes") *value = true;
      else *value = false;
   }
   
   void get_value(Dataset_t *value) {
      std::cin.clear();
      int input;
      while (1) {
         std::string input_buffer;
         getline(std::cin, input_buffer);
         std::stringstream out;
         out << input_buffer;
         if (input_buffer == "") {
            input = 0;
            break;
         }
         if (out >> input && input > 0 && input < 5) break;
         std::cout << "Invalid input" << std::endl;
         std::cin.clear();
      }
      switch (input) {
         case 1: *value = MNIST; return;
         case 2: *value = SSL_VIS; return;
         case 3: *value = VOL_VIS; return;
         case 4: *value = AOD; return;
         default: return;
      }
   }
   
   void get_value(Layer_t *value) {
      std::cin.clear();
      int input;
      while (1) {
         std::string input_buffer;
         getline(std::cin, input_buffer);
         std::stringstream out;
         out << input_buffer;
         if (input_buffer == "") {
            input = 0;
            break;
         }
         if (out >> input && input > 0 && input < 5) break;
         std::cout << "Invalid input" << std::endl;
         std::cin.clear();
      }
      switch (input) {
         case 1: *value = SIGMOID; return;
         case 2: *value = RELU; return;
         case 3: *value = GAUSSIAN; return;
         case 4: *value = SOFTMAX; return;
         default: return;
      }
   }
   
   template<typename T>
   void get_value(T *value) {
      std::cin.clear();
      T input;
      while (1) {
         std::string input_buffer;
         getline(std::cin, input_buffer);
         std::stringstream out;
         out << input_buffer;
         if (input_buffer == "") {
            return;
         }
         if (out >> input) break;
         std::cout << "Invalid input" << std::endl;
         std::cin.clear();
      }
      *value = input;
   }

   
   void get_dataset(Dataset_t *answer) {
      *answer = dataset;
      print_dataset_Q();
      print_default(*answer);
      get_value(answer);
      dataset = *answer;
      switch (*answer) {
         case MNIST:   load_MNIST_defaults();   break;
         case SSL_VIS: load_SSL_VIS_defaults(); break;
         case VOL_VIS: load_VOL_VIS_defaults(); break;
         case AOD:     load_AOD_defaults();     break;
         default: break;
      }
   }
   
   void get_stim(bool *answer) {
      *answer = multimodal;
      print_stim_query();
      print_default(*answer);
      get_value(answer);
   }
   
   void get_hidden_layer_number(int *answer) {
      *answer = hidden_layer_number;
      print_number_layers_Q();
      print_default(*answer);
      get_value(answer);
   }
   
   void get_hidden_layer_type(int i, Layer_t *answer) {
      *answer = hidden_layer_type;
      print_layer_type_Q(i);
      print_default(*answer);
      get_value(answer);
      switch (*answer) {
         case SIGMOID:  load_SIGMOID_defaults();   break;
         case GAUSSIAN: load_GAUSSIAN_defaults();  break;
         case RELU:     load_RELU_defaults();      break;
         case SOFTMAX:  load_SOFTMAX_defaults();   break;
      }
   }
   
   void get_node_number(int *answer) {
      *answer = nodenum;
      print_node_number_Q();
      print_default(*answer);
      get_value(answer);
   }
   
   void get_learning_rate(float *answer) {
      *answer = learning_rate;
      print_learning_rate_Q();
      print_default(*answer);
      get_value(answer);
   }
   
   void get_decay_rate(float *answer) {
      *answer = decay;
      print_decay_rate_Q();
      print_default(*answer);
      get_value(answer);
   }
   
   void get_dropout(bool *answer) {
      *answer = dropout;
      print_dropout_Q();
      print_default(*answer);
      get_value(answer);
   }
   
   void get_momentum(float *answer) {
      *answer = momentum;
      print_momentum_Q();
      print_default(*answer);
      get_value(answer);
   }
   
   void get_k(int *answer) {
      *answer = k;
      print_k_Q();
      print_default(*answer);
      get_value(answer);
   }
   
   void get_batchsize(int *answer) {
      *answer = batchsize;
      print_batchsize_Q();
      print_default(*answer);
      get_value(answer);
   }
   
   void get_epochs(int *answer) {
      *answer = epochs;
      print_epochs_Q();
      print_default(*answer);
      get_value(answer);
   }
   
   void get_visualization(bool *answer) {
      *answer = visualization;
      print_visualization_Q();
      print_default(*answer);
      get_value(answer);
   }
};

void add_level(MLP &mlp, Level &prev_level, Value_Handler handler, Layer *data_layer, int l, int batchsize, int valid_size, gsl_rng *rand, bool ask_stim) {
   std::cout << "Level " << l << std::endl;
   Level level;
   int node_input;
   
   node_input = handler.nodenum;
   
   if (l == 0) {
      Layer *hidden;
      switch (data_layer->type) {
         case SIGMOID: case SOFTMAX: hidden = new SigmoidLayer(node_input, batchsize, valid_size); break;
         case RELU: case GAUSSIAN: hidden = new ReLULayer(node_input, batchsize, valid_size); break;
         default: break;
      }
      std::cout << "Adding " << *hidden << std::endl;
      Connection *connection = new Connection(data_layer, hidden);
      connection->decay_rate = handler.decay;
      connection->apply_momentum = true;
      level.add(*connection);
   }
   else if(l < handler.hidden_layer_number-1) {
      for (auto layer:prev_level.top_layers) {
         Layer *hidden;
         switch (layer->type) {
            case SIGMOID: case SOFTMAX: hidden = new SigmoidLayer(node_input, batchsize, valid_size); break;
            case RELU: case GAUSSIAN: hidden = new ReLULayer(node_input, batchsize, valid_size); break;
            default: break;
         }
         std::cout << "Adding " << *hidden << std::endl;
         Connection *connection = new Connection(layer, hidden);
         connection->decay_rate = handler.decay;
         connection->apply_momentum = true;
         level.add(*connection);
      }
   }
   else {
      Layer *hidden;
      if (prev_level.top_layers.size() == 1 && prev_level.top_layers[0]->type == SIGMOID)
         hidden = new SigmoidLayer(node_input, batchsize, valid_size);
      else hidden = new ReLULayer(node_input, batchsize, valid_size);
      std::cout << "Adding " << *hidden << std::endl;
      for (auto layer:prev_level.top_layers) {
         Connection *connection = new Connection(layer,hidden);
         connection->decay_rate = handler.decay;
         connection->apply_momentum = true;
         level.add(*connection);
      }
   }
   
   bool stim = false;
   if (ask_stim) handler.get_stim(&stim);
   if (stim || (l == handler.hidden_layer_number -1 && handler.multimodal)) {
      DataSet *stim_data;
      Layer *stim_layer;
      std::cout << "Adding label data" << std::endl;
      switch (handler.dataset) {
         case MNIST: {
            stim_data = load_MNIST_L();
            stim_data->make_validation(gsl_rng_clone(rand));
            std::cout << "Loading MNIST lables" << std::endl;
            stim_layer = new SoftmaxLayer(stim_data->data.dim2, batchsize, stim_data->validation.dim1);
            break;
         }
         case SSL_VIS: case VOL_VIS: {
            std::cout << "Loading visuomotor fMRI stim" << std::endl;
            stim_data = load_FMRI_S();
            stim_data->make_validation(gsl_rng_clone(rand));
            stim_layer = new SigmoidLayer(stim_data->data.dim2, batchsize, stim_data->validation.dim1);
            break;
         }
         case AOD: {
            std::cout << "Loading AOD stim" << std::endl;
            stim_data = load_AOD_stim(RUN1, data_layer->data->dims[3]/249);
            stim_data->make_validation(gsl_rng_clone(rand));
            stim_layer = new SigmoidLayer(stim_data->data.dim2, batchsize, stim_data->validation.dim1);
            break;
         }
         default: exit(EXIT_FAILURE);
      }
      std::cout << "Adding " << *stim_layer << std::endl;
      stim_layer->data = stim_data;
      mlp.data_layers.push_back(stim_layer);
      
      Layer *hidden;
      if (l < handler.hidden_layer_number -1) {
         switch (stim_layer->type) {
            case SIGMOID: case SOFTMAX: hidden = new SigmoidLayer(node_input, batchsize, valid_size); break;
            case RELU: case GAUSSIAN: hidden = new ReLULayer(node_input, batchsize, valid_size); break;
            default: break;
         }
         std::cout << "Adding " << *hidden << std::endl;
      }
      else hidden = level.top_layers[0];
      
      hidden->learning_rate = 0;
      Connection *connection = new Connection(stim_layer, hidden);
      connection->decay_rate = handler.decay;
      level.add(*connection);
      connection->transmission_scale = 10;
      connection->learning_rate = 0.00001;
      connection->weight_max_length = 2;
      connection->decay_rate = AUTO;
   }
   //handler.nodenum/=2;
   mlp.add(level);
   prev_level = level;
}

DBN *return_network() {
   
   MLP mlp;
   Value_Handler handler;
   
   Dataset_t data_input;
   handler.get_dataset(&data_input);
   
   int layer_number, batchsize;
#if 0
   handler.get_batchsize(&batchsize);
#else
   batchsize = handler.batchsize;
#endif
   gsl_rng *rand = gsl_rng_clone(r);
   
   Layer *data_layer;
   DataSet *data;
   switch (data_input) {
      case MNIST: {
         data = load_MNIST_DS();
         data->make_validation(gsl_rng_clone(rand));
         std::cout << "Loading MNIST with Sigmoid data layer" << std::endl;
         data_layer = new SigmoidLayer(data->data.dim2, batchsize, data->validation.dim1);
         break;
      }
      case SSL_VIS: {
         std::cout << "Loading multiple subject Visuomotor with Gaussian data layer" << std::endl;
         data = load_SS_fMRI_DS();
         data->make_validation(gsl_rng_clone(rand));
         data_layer = new GaussianLayer(data->data.dim2, batchsize, data->validation.dim1);
         break;
      }
      case VOL_VIS: {
         std::cout << "Loading single subject 3D Visuomotor with Gaussian data layer" << std::endl;
         data = load_fMRI3D_DS();
         data->make_validation(gsl_rng_clone(rand));
         data_layer = new GaussianLayer(data->data.dim2, batchsize, data->validation.dim1);
         break;
      }
      case AOD: {
         std::cout << "Loading AOD with Gaussian data layer" << std::endl;
         data = load_fMRI3D_DS(aod_path);
         data->make_validation(gsl_rng_clone(rand));
         data_layer = new GaussianLayer(data->data.dim2, batchsize, data->validation.dim1);
         break;
      }
      default: exit(EXIT_FAILURE);
   }
   data_layer->data = data;
   mlp.viz_layer = data_layer;
   mlp.data_layers.push_back(data_layer);
   
   handler.get_hidden_layer_number(&layer_number);
   Level prev_level;
   for (int l = 0; l < layer_number; ++l) { add_level(mlp, prev_level, handler, data_layer, l, batchsize, (int)data->validation.dim1, rand);}
   
   DBN *dbn = new DBN(mlp);
   
   std::cout << "Created DBN (or RBM in the case of one hidden layer)" << std::endl;
   
   std::cout << "Constructing cd learner" << std::endl;
   float momentum;
   int epochs, k;
   momentum = handler.momentum;
   k = handler.k;
   epochs = handler.epochs;
   bool viz_input;
   handler.get_visualization(&viz_input);
   
   if (viz_input) {
      dbn->monitor_dbn = true;
   }
   std::cout << "Finished created DBN" << std::endl;
   
   dbn->name = "loaded" + convert_to_string(data->type) + ".";
   dbn->name += ".dbn";
   std::cout << mlp;
   return dbn;
}

DBN *return_aod_network(std::string filename, std::ofstream &log_stream, std::streambuf *psbuf) {

   Value_Handler handler;
   gsl_rng *rand = gsl_rng_clone(r);
   
   handler.load_AOD_defaults();
   std::ifstream infile;
   std::string file = path+filename;
   std::cout << "Opening config file " << file.c_str() << std::endl;
   infile.open(file.c_str());
   
   if (!infile.is_open()) error("Couldn't open config file");
   
   std::string h5path;
   std::string h5file;
   std::string line;
   
   while (getline(infile, line)) {
      std::string       tag, value;
      std::stringstream linestream;
      linestream << line;
      std::getline(linestream, tag, ' ');
      std::getline(linestream, value, '\n');
      std::cout << tag << ": " << value << std::endl;
      if (tag == "path") h5path = value;
      else if (tag == "filename") h5file = value;
      else if (tag == "layers") handler.hidden_layer_number = atoi(value.c_str());
      else if (tag == "nodenum") handler.nodenum = atoi(value.c_str());
      else if (tag == "batchsize") handler.batchsize = atoi(value.c_str());
      else if (tag == "k") handler.k = atoi(value.c_str());
      else if (tag == "bottleneck") handler.bottleneck = atof(value.c_str());
      else if (tag == "momentum") handler.momentum = atof(value.c_str());
      else if (tag == "stim") handler.multimodal = true;
      else if (tag == "epochs") handler.epochs = atoi(value.c_str());
      else {std::cout << "Warning, unknown command found: " << tag << std::endl;}
   }
   infile.close();
   if (h5file.size() == 0) error("No file specified");
   //handler.load_VOL_fMRI_defaults();
   MLP mlp;
   
   std::cout << "Loading data" << std::endl;
   DataSet *data = load_fMRI3D_DS(h5path+h5file);
   data->make_validation(gsl_rng_clone(rand));
   std::cout << "Loaded " << h5file << std::endl;
   
   GaussianLayer *data_layer = new GaussianLayer(data->data.dim2, handler.batchsize, data->validation.dim1);
   data_layer->data = data;
   mlp.viz_layer = data_layer;
   mlp.data_layers.push_back(data_layer);
   
   Level prev_level;
   std::string nodes_str = "";
   for (int l = 0; l < handler.hidden_layer_number; ++l) {
      nodes_str += convert_to_string(handler.nodenum) + "|";
      add_level(mlp, prev_level, handler, data_layer, l, handler.batchsize, (int)data->validation.dim1, rand, false);
      handler.nodenum*=handler.bottleneck;
   }
   DBN *dbn = new DBN(mlp);
   
   
   unsigned long u = gsl_rng_uniform_int(r,1000);
   out_path += convert_to_string(u) + "/";
   mkdir((out_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
   
   dbn->name = h5file;
   dbn->name += ".l=" + convert_to_string(handler.hidden_layer_number);
   if (handler.multimodal) dbn->name += ".+s";
   dbn->name += ".n=" + nodes_str;
   dbn->name += ".dbn";
   
   log_stream.open((out_path + dbn->name + ".log").c_str());
   psbuf = log_stream.rdbuf();
   std::cout.rdbuf(psbuf);
   std::cerr.rdbuf(psbuf);
   
   
   std::cout << "DBN constructued" << std::endl;
   handler.print();
   std::cout << *dbn;
   return dbn;
}

DBN *load_and_stack(std::string filename, std::ofstream &log_stream, std::streambuf *psbuf) {
   MLP mlp = load_MLP(filename);
   Value_Handler handler;
   gsl_rng *rand = gsl_rng_clone(r);
   
   
   handler.load_AOD_defaults();
   Level top_level = mlp.levels.back();
   Layer *top_layer = top_level.top_layers[0];
   Layer *data_layer = mlp.viz_layer;
   DataSet *data = data_layer->data;
   handler.nodenum = (int)top_layer->nodenum;
   handler.batchsize = (int)top_layer->m_learning.dim1;
   
   //tmp
   Layer *bot_layer = mlp.levels[0].bot_layers[0];
   bot_layer->rec_biases = bot_layer->biases;
   top_layer->rec_biases = top_layer->biases;
   //
   
   for (int i = 0; i < mlp.levels.size()-1; ++i) {mlp.levels[i].transport_data(mlp.levels[i+1]);}
   
   DBN *dbn = new DBN(mlp);
   
   for (auto level:dbn->reference_MLP.levels) dbn->add(level);
   add_level(dbn->reference_MLP, top_level, handler, data_layer, (int)mlp.levels.size(), handler.batchsize, (int)data->validation.dim1, rand, false);
   dbn->reference_MLP.levels[dbn->reference_MLP.levels.size()-2].transport_data(dbn->reference_MLP.levels.back());
   
   dbn->name = filename;
   dbn->name += ".+l.dbn";
   
   log_stream.open((out_path + dbn->name + ".log").c_str());
   psbuf = log_stream.rdbuf();
   std::cout.rdbuf(psbuf);
   std::cerr.rdbuf(psbuf);
   
   std::cout << "DBN constructued" << std::endl;
   handler.print();
   std::cout << *dbn;
   return dbn;
}