//
//  main.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/10/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "RBM.h"
#include "Viz.h"
#include "IO.h"
#include "Types.h"
#include "Timecourses.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_permute.h"

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
   
   DataSet data1,data2;
   data1.loadfMRI();
   data2.loadstim();
   //---------DONE INIT
   //--------- GSL TESTS GO HERE ----------------
   
   print_gsl(data2.train);
   
   //INIT RBM
   
   GaussianLayer baselayer((int)data1.train->size2);
   SigmoidLayer stimuluslayer((int)data2.train->size2);
   ReLULayer hiddenlayer(10);
   
   Connection c1(&baselayer, &hiddenlayer);
   c1.decay_ = 0.0002;
   c1.learning_rate_ = 0.000008;
   Connection c2(&stimuluslayer, &hiddenlayer);
   c2.decay_ = 0.0002;
   c2.learning_rate_ = 0.008;
   
   RBM rbm(&c1, &c2);
   
   ReLULayer h2(10);
   
   Connection c3(&hiddenlayer, &h2);
   RBM rbm2(&c3);
   
   rbm.load_DS(&data1, &data2);
   
   //--------------
   float momentum = 0.5;
   float k = 1;
   float sparsitytarget = 0.1;
   float decayrate = 0.9;
   float sparsitycost = 0;
   float batchsize = 1;
   
   //LEARNING!!!!!!!!!
   ContrastiveDivergence cdLearner(&rbm, momentum, k, sparsitytarget, decayrate, sparsitycost, batchsize);
   
   for (int epoch = 1; epoch < 300; ++epoch){
      std::cout << "Epoch " << epoch << std::endl;
      cdLearner.run();
      //cdLearner.learningRate_ *= .95;
   }
   
   print_gsl(c1.top_->biases_);
   
   std::cout << "Done Learning! " << std::endl;
   
   get_timecourses(&rbm, &data1);
   
   rbm.visualize(0, 1);
    
   
   return 0; 
}

