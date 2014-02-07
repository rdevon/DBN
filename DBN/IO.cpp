//
//  IO.cpp
//  DBN
//
//  Created by Devon Hjelm on 11/13/12.
//
//

#include "IO.h"
#include "MLP.h"
#include "Connections.h"
#include "Layers.h"
#include "DataSets.h"
#include "DBN.h"
#include "Monitors.h"
#include "Monitor_Units.h"
#include "Viz_Units.h"
#include <string.h>
#include "SupportFunctions.h"

herr_t file_info(hid_t loc_id, const char *name, void *list) {
   std::vector<std::string> *listcast = (std::vector<std::string>*)list;
   std::stringstream out;
   out << name;
   std::string strname = out.str();
   listcast->push_back(name);
   return 0;
}

void save(const MLP& mlp) {
   typedef DataSet DBN_DS;
   using namespace H5;
   using H5::DataSet;
   
   const H5std_string   FILE_NAME(out_path + mlp.name + ".h5");
   H5File file(FILE_NAME, H5F_ACC_TRUNC);
   Group layers = file.createGroup("Layers");
   Group levels = file.createGroup("Levels");
   
   save(mlp.layers, layers);
   layers.close();
   
   hsize_t dl_dims[1] = {mlp.data_layers.size()};
   int dl_lyr_buff[mlp.data_layers.size()];
   for (int i = 0; i < mlp.data_layers.size(); ++i) {
      Layer *data_layer = mlp.data_layers[i];
      int dlidx = get_index(data_layer, mlp.layers);
      dl_lyr_buff[i] = dlidx;
   }
   
   DataSet data_layers = file.createDataSet("data_layers", PredType::NATIVE_INT, DataSpace(1, dl_dims));
   data_layers.write(&dl_lyr_buff, PredType::NATIVE_INT);
   
   if (mlp.viz_layer) {
      Attribute viz_att = data_layers.createAttribute("viz", PredType::NATIVE_INT, DataSpace(H5S_SCALAR));
      int viz_idx = get_index(mlp.viz_layer, mlp.layers);
      viz_att.write(PredType::NATIVE_INT, &viz_idx);
      viz_att.close();
   }
   
   data_layers.close();
   
   int l = 0;
   for (auto level:mlp.levels) {
      Group lev = levels.createGroup(convert_to_string(l));
      save(level, mlp.layers, lev);
      lev.close();
      ++l;
   }
}

MLP load_MLP(std::string file_name) {
   typedef DataSet DBN_DS;
   using namespace H5;
   using H5::DataSet;
   
   MLP mlp;
   H5File file(out_path + file_name, H5F_ACC_RDONLY);
   Group layers = file.openGroup("Layers");
   Group levels = file.openGroup("Levels");
   
   mlp.layers = load_layers(layers);
   layers.close();
   
   DataSet data_layers = file.openDataSet("data_layers");
   DataSpace dl_dsp = data_layers.getSpace();
   hsize_t dims[1];
   dl_dsp.getSimpleExtentDims(dims);
   int data_layer_ids [dims[0]];
   data_layers.read(&data_layer_ids, PredType::NATIVE_INT);
   
   gsl_rng *rand = gsl_rng_clone(r);
   for (int i = 0; i < dims[0]; ++i) {
      Layer *data_layer = mlp.layers[data_layer_ids[i]];
      mlp.data_layers.push_back(data_layer);
      data_layer->data->make_validation(gsl_rng_clone(rand));
   }
   
   try {
      Attribute viz_att = data_layers.openAttribute("viz");
      int viz_idx;
      viz_att.read(PredType::NATIVE_INT, &viz_idx);
      mlp.viz_layer = mlp.layers[viz_idx];
   } catch (Exception E) {}
   
   data_layers.close();
      
   std::vector<std::string> lids;
   file.iterateElems("/Levels/", NULL, file_info, &lids);
   for (auto lidx:lids) {
      Group level_group = levels.openGroup(lidx);
      Level level = load_level(level_group, mlp.layers);
      mlp.add(level);
   }
   return mlp;
}

//------Save/Load Levels

void save(const Level& level, const std::vector<Layer*>& layers, H5::Group& group) {
   typedef DataSet DS;
   using namespace H5;
   using H5::DataSet;
   int i = 0;
   for (auto connection:level.connections) {
      hsize_t dims[2];
      dims[0] = connection->weights.dim1;
      dims[1] = connection->weights.dim2;
      DataSpace dsp(2, dims);
      DataSet ds = group.createDataSet(convert_to_string(i), PredType::NATIVE_FLOAT, dsp);
      save(*connection, ds);
      hsize_t att_dims[1] = {2};
      Attribute layer_ids_att = ds.createAttribute("layer_ids", PredType::NATIVE_INT, DataSpace(1,att_dims));
      
      int layer_ids[2] = {get_index(connection->from, layers),get_index(connection->to, layers)};
      layer_ids_att.write(PredType::NATIVE_INT, &layer_ids[0]);
      layer_ids_att.close();
      
      dsp.close();
      ds.close();
      ++i;
   }
}

Level load_level(H5::Group& group, std::vector<Layer*>& layers) {
   typedef DataSet DS;
   using namespace H5;
   using H5::DataSet;
   
   Level level;
   for (int i = 0; i < group.getNumObjs(); ++i) {
      H5std_string d_name = group.getObjnameByIdx(i);
      DataSet ds = group.openDataSet(d_name);
      Attribute lids_att = ds.openAttribute("layer_ids");
      int l_ids[2];
      lids_att.read(PredType::NATIVE_INT, &l_ids);
      lids_att.close();
      
      Layer *from = layers[l_ids[0]];
      Layer *to = layers[l_ids[1]];
      Connection *connection = load_connection(ds, from, to);
      level.add(*connection);
      ds.close();
   }
   return level;
}

//------Save/Load Connections

void save(const std::vector<Connection*>& connections, const std::vector<Layer*>& layers, H5::Group& group) {
   typedef DataSet DS;
   using namespace H5;
   using H5::DataSet;
   int i = 0;
   for (auto connection:connections) {
      hsize_t dims[2];
      dims[0] = connection->weights.dim1;
      dims[1] = connection->weights.dim2;
      DataSpace dsp(2, dims);
      DataSet ds = group.createDataSet(convert_to_string(i), PredType::NATIVE_FLOAT, dsp);
      save(*connection, ds);
      Attribute layer_ids_att = ds.createAttribute("layer_ids", PredType::NATIVE_INT, DataSpace(H5S_SCALAR));
      
      int layer_ids[2] = {get_index(connection->from, layers),get_index(connection->to, layers)};
      layer_ids_att.write(PredType::NATIVE_INT, &layer_ids);
      layer_ids_att.close();
      
      dsp.close();
      ds.close();
      ++i;
   }
}

std::vector<Connection*> load_connections(H5::Group& group, std::vector<Layer*>& layers) {
   typedef DataSet DS;
   using namespace H5;
   using H5::DataSet;
   
   std::vector<Connection*> connections;
   for (int i = 0; i < group.getNumObjs(); ++i) {
      H5std_string d_name = group.getObjnameByIdx(i);
      DataSet ds = group.openDataSet(d_name);
      Attribute lids_att = ds.openAttribute("layer_ids");
      int l_ids[2];
      lids_att.read(PredType::NATIVE_INT, &l_ids);
      lids_att.close();
      
      Layer *from = layers[l_ids[0]];
      Layer *to = layers[l_ids[1]];
      Connection *connection = load_connection(ds, from, to);
      connections.push_back(connection);
      ds.close();
   }
   return connections;
}

void save(const Connection& connection, H5::DataSet &ds) {
   typedef DataSet DS;
   using namespace H5;
   using H5::DataSet;
   
   ds.write(connection.weights.m->data, PredType::NATIVE_FLOAT);
}

Connection *load_connection(H5::DataSet& ds, Layer* from, Layer* to) {
   typedef DataSet DS;
   using namespace H5;
   using H5::DataSet;
   
   Connection *connection = new Connection(from, to);
   ds.read(connection->weights.m->data, PredType::NATIVE_FLOAT);
   return connection;
}

//------Save/Load Layers

void save(const std::vector<Layer*>& layers, H5::Group& group) {
   typedef DataSet DS;
   using namespace H5;
   using H5::DataSet;
   
   int i = 0;
   for (auto layer:layers) {
      hsize_t dims[1];
      dims[0] = layer->nodenum;
      DataSpace dsp(1,dims);
      DataSet ds = group.createDataSet(convert_to_string(i), PredType::NATIVE_FLOAT, dsp);
      save(*layer,ds);
      ds.close();
      dsp.close();
      ++i;
   }
}

std::vector<Layer*> load_layers(H5::Group &group) {
   typedef DataSet DS;
   using namespace H5;
   using H5::DataSet;
   std::vector<Layer*> layers;
   int obj_num = (int)group.getNumObjs();
   for (int i = 0; i < obj_num; ++i) {
      H5std_string d_name = group.getObjnameByIdx(i);
      DataSet ds = group.openDataSet(d_name);
      Layer *layer = load_layer(ds);
      layers.push_back(layer);
   }
   return layers;
}

void save(const Layer& layer, H5::DataSet &ds) {
   
   typedef DataSet DS;
   using namespace H5;
   using H5::DataSet;
   
   ds.write(layer.biases.m->data, PredType::NATIVE_FLOAT);
   
   Attribute type = ds.createAttribute("type", PredType::NATIVE_INT, DataSpace(H5S_SCALAR));
   type.write(PredType::NATIVE_INT, &layer.type);
   Attribute batch_size = ds.createAttribute("batch_size", PredType::NATIVE_INT, DataSpace(H5S_SCALAR));
   batch_size.write(PredType::NATIVE_INT, &layer.m_learning.dim1);
   Attribute test_size = ds.createAttribute("test_size", PredType::NATIVE_INT, DataSpace(H5S_SCALAR));
   test_size.write(PredType::NATIVE_INT, &layer.m_testing.dim1);
   type.close();
   if (layer.data) {
      Attribute data_type = ds.createAttribute("data_type", PredType::NATIVE_INT, DataSpace(H5S_SCALAR));
      data_type.write(PredType::NATIVE_INT, &layer.data->type);
      data_type.close();
      StrType vlst(PredType::C_S1, 256);
      H5std_string strwritebuf(layer.data->data_path);
      Attribute data_path = ds.createAttribute("data_path", vlst, DataSpace(H5S_SCALAR));
      data_path.write(vlst, strwritebuf);
      data_path.close();
   }
}

Layer *load_layer(H5::DataSet&ds) {
   
   typedef DataSet DS;
   using namespace H5;
   using H5::DataSet;
   
   DataSpace dsp = ds.getSpace();
   hsize_t dims[1];
   dsp.getSimpleExtentDims(dims);
   int size = (int)dims[0];
   int batch_size, test_size;
   Attribute type_att = ds.openAttribute("type");
   Layer_t type;
   type_att.read(PredType::NATIVE_INT, &type);
   Attribute batch_size_att = ds.openAttribute("batch_size");
   batch_size_att.read(PredType::NATIVE_INT, &batch_size);
   Attribute test_size_att = ds.openAttribute("test_size");
   test_size_att.read(PredType::NATIVE_INT, &test_size);
   type_att.close();
   batch_size_att.close();
   test_size_att.close();
   
   Layer *layer;
   std::cout << "   Loading ";
   switch (type) {
      case SIGMOID: {
         layer = new SigmoidLayer(size,batch_size,test_size);
         std::cout << "sigmoid layer with " << size << " nodes" << std::endl;
         break;
      }
      case GAUSSIAN: {
         layer = new GaussianLayer(size,batch_size,test_size);
         std::cout << "gaussian layer with " << size << " nodes" << std::endl;
         break;
      }
      case RELU: {
         layer = new ReLULayer(size,batch_size,test_size);
         std::cout << "rectified linear layer with " << size << " nodes" << std::endl;
         break;
      }
      case SOFTMAX: {
         layer = new SoftmaxLayer(size,batch_size,test_size);
         std::cout << "softmax layer with " << size << " nodes" << std::endl;
         break;
      }
   }
   ds.read(layer->biases.m->data, PredType::NATIVE_FLOAT);
   
   try {
      std::cout << "***If error, ignore output" << std::endl;
      Dataset_t data_type;
      std::stringstream data_path;
      Attribute data_type_att = ds.openAttribute("data_type");
      Attribute data_path_att = ds.openAttribute("data_path");
      data_type_att.read(PredType::NATIVE_INT, &data_type);
      data_type_att.close();
      
      H5std_string strreadbuf ("");
      data_path_att.read(StrType(PredType::C_S1, 256), strreadbuf);
      data_path << strreadbuf;
      data_path_att.close();
      switch (data_type) {
         case MNIST: layer->data = load_MNIST_DS(data_path.str());  break;
         case MNIST_L: layer->data = load_MNIST_L(data_path.str());  break;
         case SSL_VIS: layer->data = load_SS_fMRI_DS(data_path.str());  break;
         case VOL_VIS: case AOD: layer->data = load_fMRI3D_DS(data_path.str()); break;
         case AOD_STIM: {
            run_t run;
            if (data_path.str() == "run1") run = RUN1;
            else if (data_path.str() == "run2") run = RUN2;
            else if (data_path.str() == "run12") run = RUN12;
            layer->data = load_AOD_stim(run);
         } break;
         default: layer->data = new DS(NA,0,0); layer->data->data_path = data_path.str(); break;
      }
   } catch(Exception E) {}
   dsp.close();
   return layer;
}

//------Save Features

void save_features(Layer_Monitor* monitor, std::string name) {
   std::cout << "Saving features in file" << name << "_features.h5" << std::endl;
   typedef DataSet DBN_DS;
   using namespace H5;
   using H5::DataSet;
   
   const H5std_string   FILE_NAME(out_path + name + "_features.h5");
   H5File file(FILE_NAME, H5F_ACC_TRUNC);
   Group features = file.createGroup("/Features");
   Group classes = file.createGroup("/Classes");
   Group timecourses = file.createGroup("/TimeCourses");
   
   hsize_t dims[1];
   dims[0] = monitor->rc_monitor->line_set.dim2;
   DataSpace rc_dsp(1,dims);
   DataSet rc_ds = file.createDataSet("reconstruction_cost", PredType::NATIVE_FLOAT, rc_dsp);
   rc_ds.write(monitor->rc_monitor->line_set.m->data, PredType::NATIVE_FLOAT);
   rc_ds.close();
   rc_dsp.close();
   
   int idx = 0;
   for (auto unit:((monitor->layer_monitor)->units)) {
      Matrix *feature = &((Stacked_Tex_Unit*)unit)->viz_matrix;
      hsize_t dims[2];
      dims[0] = feature->dim1;
      dims[1] = feature->dim2;
      DataSpace dsp(2, dims);
      DataSet ds = features.createDataSet(convert_to_string(idx), PredType::NATIVE_FLOAT, dsp);
      ds.write(feature->m->data, PredType::NATIVE_FLOAT);
      ds.close();
      dsp.close();
      ++idx;
   }
   
   idx = 0;
   if (monitor->class_monitor != NULL) {
      for (auto unit:((monitor->class_monitor)->units)) {
         Matrix *feature = &((Stacked_Tex_Unit*)unit)->viz_matrix;
         hsize_t dims[2];
         dims[0] = feature->dim1;
         dims[1] = feature->dim2;
         DataSpace dsp(2, dims);
         DataSet ds = classes.createDataSet(convert_to_string(idx), PredType::NATIVE_FLOAT, dsp);
         ds.write(feature->m->data, PredType::NATIVE_FLOAT);
         ds.close();
         dsp.close();
         ++idx;
      }
   }
#if 0
   idx = 0;
   for (auto unit:((monitor->timecourse_monitor)->units)) {
      gsl_vector_float *timecourse = ((Plot_Unit*)unit)->line_set;
      hsize_t dims[1];
      dims[0] = timecourse->size;
      DataSpace dsp(1, dims);
      DataSet ds = timecourses.createDataSet(convert_to_string(idx), PredType::NATIVE_FLOAT, dsp);
      ds.write(timecourse->data, PredType::NATIVE_FLOAT);
      ds.close();
      dsp.close();
      ++idx;
   }
#endif

}