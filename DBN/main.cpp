//
//  main.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/10/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "Types.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_permute.h"
#include "IO.h"
#include "Layers.h"
#include "Connections.h"
#include "RBM.h"
#include "Viz.h"

int main (int argc, const char * argv[])
{
   //--------------RNG INIT STUFF
   
   srand((unsigned)time(0));
   
   long seed;
   r = gsl_rng_alloc (gsl_rng_rand48);     // pick random number generator
   seed = time (NULL) * getpid();
   gsl_rng_set (r, seed);                  // set seed
   //--------------
   //LOAD DATASET and INIT
   
   DataSet data1;
   data1.loadMNIST();
   //---------DONE INIT
   //--------- GSL TESTS GO HERE ----------------
   
   //INIT RBM
   
   SigmoidLayer baselayer((int)data1.train->size2);
   baselayer.shapeInput(&data1);
   //SigmoidLayer stimuluslayer((int)data2.train->size2);
   SigmoidLayer hiddenlayer(100);
   //ReLULayer hiddenlayer2(16);
   
   InputEdge ie1(&data1);
   //InputEdge ie2(&data2);
   
   baselayer.input_edge = &ie1;
   //stimuluslayer.input_edge = &ie2;
   
   Connection c1(&baselayer, &hiddenlayer);
   c1.learning_rate = 0.01;
   c1.decay = 0;
   //Connection c2(&stimuluslayer, &hiddenlayer);
   //c2.learning_rate = 0.0000008;
   //c2.decay = 0.000002;
   
   //Connection c3(&hiddenlayer, &hiddenlayer2);
   //c3.learning_rate = 0.00001;
   //c3.decay = 0.000001;
   
   //--------------
   float momentum = 0;
   float k = 1;
   float batchsize = 10;
   float epochs = 500;
   
   //LEARNING!!!!!!!!!
   ContrastiveDivergence cdLearner(momentum, k, batchsize, epochs);
   Connection_Learning_Monitor monitor(&c1);
   cdLearner.monitor = &monitor;
   
   RBM rbm;
   rbm.add_connection(&c1);
   rbm.teacher = &cdLearner;
   rbm.learn();
   
   /*
   DBN dbn;
   dbn.add_connection(&c1);
   dbn.add_connection(&c3);
   dbn.teacher = &cdLearner;
   
   dbn.finish_setup();
   
   dbn.pathways[0]->setup_viz();
   
   dbn.viz = (dbn.pathways[0])->viz;
   dbn.viz->initViz();
   dbn.viz->thresh = .02;
   
   dbn.viz->scale = 1;
   dbn.learn();
   
   std::cout << "Done Learning! " << std::endl;
    */
   
   return 0; 
}

