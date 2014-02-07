//
//  Tests.cpp
//  DBN
//
//  Created by Devon Hjelm on 1/3/13.
//
//

#include "Tests.h"
#include "Layers.h"
#include "Connections.h"
#include "DataSets.h"
#include "RBM.h"
#include "DBN.h"
#include "Autoencoder.h"
#include "IO.h"

void transport_data() {
   SigmoidLayer sig1(6,2,1);
   ReLULayer sig2(4,2,1);
   ReLULayer sig3(4,2,1);
   DataSet ds(NA,5,3,2,1);
   Matrix data(5,6);
   for (int i = 0; i < data.dim1*data.dim2; ++i) *(data.m->data+i) = i%4;
   ds.data = data;
   Connection conn1(&sig1, &sig2);
   conn1.weights.set_all(2);
   Connection conn2(&sig2, &sig3);
   Level rbm1;
   rbm1.add(conn1);
   Level rbm2;
   rbm2.add(conn2);
   rbm1.transport_data(rbm2);
   std::cout << data << conn1.weights;
   std::cout << rbm2.top_layers[0]->data->data;
   
   exit(EXIT_SUCCESS);
}

void test_ae() {
   SigmoidLayer l1(4,4,1);
   SigmoidLayer l2(2,4,1);
   Connection c(&l1,&l2);
   c.weights.set_gaussian();
   DataSet ds(NA, 4,0,0,0,4);
   Matrix data(4,4);
   data.set_gaussian();
   ds.data = data;
   ds.train = data;
   
   exit(1);
}

void test_mlp() {
   SigmoidLayer sig1(6,2,1);
   ReLULayer relu1(4,2,1);
   ReLULayer relu2(4,2,1);
   DataSet ds(NA,5,3,2,1);
   Matrix data(5,6);
   for (int i = 0; i < data.dim1*data.dim2; ++i) *(data.m->data+i) = i%4;
   ds.train = data;
   sig1.data = &ds;
   std::cout << sig1.data->train;
   Connection conn1(&sig1, &relu1);
   conn1.weights.set_all(1);
   Connection conn2(&relu1, &relu2);
   
   Level l1, l2;
   l1.add(conn1);
   l2.add(conn2);
   
   MLP mlp;
   mlp.add(l1);
   mlp.add(l2);
   
   mlp.init_data();
   mlp.pull_data(LEARNING);
   mlp.transmit(UP, LEARNING, NOSAMPLE);
   std::cout << sig1.m_learning << relu1.m_learning << relu2.m_learning;
   
   exit(1);
}

void test_save_load() {
   SigmoidLayer sig1(6,2,1);
   for (int i = 0; i < sig1.biases.dim1*sig1.biases.dim2; ++i) *(sig1.biases.m->data+i) = i%3;
   
   SigmoidLayer sig2(7,2,1);
   for (int i = 0; i < sig2.biases.dim1*sig2.biases.dim2; ++i) *(sig2.biases.m->data+i) = i%3;
   
   ReLULayer relu1(4,2,1);
   for (int i = 0; i < relu1.biases.dim1*relu1.biases.dim2; ++i) *(relu1.biases.m->data+i) = i%4;
   ReLULayer relu2(4,2,1);
   for (int i = 0; i < relu2.biases.dim1*relu2.biases.dim2; ++i) *(relu2.biases.m->data+i) = i%5;
   DataSet ds(NA,5,3,2,1);
   Matrix data(5,6);
   for (int i = 0; i < data.dim1*data.dim2; ++i) *(data.m->data+i) = i%6;
   ds.train = data;
   ds.data_path = "testpath";
   sig1.data = &ds;
   std::cout << sig1.data->train;
   Connection conn1(&sig1, &relu1);
   for (int i = 0; i < conn1.weights.dim1*conn1.weights.dim2; ++i) *(conn1.weights.m->data+i) = i%7;
   Connection conn3(&sig2, &relu1);
   for (int i = 0; i < conn3.weights.dim1*conn3.weights.dim2; ++i) *(conn3.weights.m->data+i) = i%7;
   Connection conn2(&relu1, &relu2);
   for (int i = 0; i < conn2.weights.dim1*conn2.weights.dim2; ++i) *(conn2.weights.m->data+i) = i%8;
   
   Level l1, l2;
   l1.add(conn1);
   l1.add(conn3);
   l2.add(conn2);
   
   MLP mlp;
   mlp.add(l1);
   mlp.add(l2);
   mlp.name = "testsave";
   mlp.data_layers.push_back(&sig1);
   save(mlp);
   MLP clone = load_MLP(out_path+"testsave.h5");
   std::cout << mlp << clone;
   for (auto layer:clone.layers) std::cout << layer->biases;
   for (auto connection:clone.connections) std::cout << connection->weights;
   std::cout << clone.data_layers[0]->data->data_path;
   
   exit(1);
}

void test_aod_stim() {
   DataSet *ds = load_AOD_stim(RUN1);
   ds->data.save("/Users/devon/Research/tmp/aod_stim_test");
   exit(1);
}

void test_copy_mat() {
   Matrix mat1(4,5);
   mat1.set_gaussian();
   Matrix mat2(mat1);
   Matrix mat3 = mat1;
   std::cout << mat1.m << " " << mat2.m << " " << mat3.m;
   exit(1);
}
