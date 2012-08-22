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
   data2.loadSPM();
   //---------DONE INIT
   
   //INIT RBM
   
   GaussianLayer baselayer((int)data1.train->size2);
   GaussianLayer stimuluslayer((int)data2.train->size2);
   ReLULayer hiddenlayer(16);
   
   Connection c1(&baselayer, &hiddenlayer);
   Connection c2(&stimuluslayer, &hiddenlayer);

   RBM rbm(&c1, &c2);
   
   rbm.load_DS(&data1, &data2);
   
   //--------------
   float learningrate = 0.000008;
   float weightcost = 0.0002;
   float momentum = 0.6;
   float k = 1;
   float sparsitytarget = 0.1;
   float decayrate = 0.9;
   float sparsitycost = 0;
   float batchsize = 1;
   
   //LEARNING!!!!!!!!!
   ContrastiveDivergence cdLearner(&rbm, learningrate, weightcost, momentum, k, sparsitytarget, decayrate, sparsitycost, batchsize);
   
   for (int epoch = 1; epoch < 200; ++epoch){
      std::cout << "Epoch " << epoch << std::endl;
      cdLearner.run();
      //cdLearner.learningRate_ *= .95;
   }
   
   get_timecourses(&rbm, &data1);
   
   //rbm.visualize(1, -1);
    
   
   return 0; 
}

