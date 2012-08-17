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
   
   DataSet data;
   data.loadfMRI();
   
   //---------DONE INIT
  
   //INIT RBM
   
   GaussianLayer baselayer((int)data.train->size2);
   ReLULayer hiddenlayer(20);
   
   Connection c1(&baselayer, &hiddenlayer);

   RBM rbm(&c1);
   
   //--------------
   float learningrate = 0.0001;
   float weightcost = 0;
   float momentum = 0;
   float k = 1;
   float sparsitytarget = 0.1;
   float decayrate = 0.9;
   float sparsitycost = 0;
   float batchsize = 1;

   baselayer.shapeInput(data.train);
   
   //LEARNING!!!!!!!!!
   ContrastiveDivergence cdLearner(&rbm, &data, learningrate, weightcost, momentum, k, sparsitytarget, decayrate, sparsitycost, batchsize);
   
   for (int epoch = 1; epoch < 100; ++epoch){
      std::cout << "Epoch " << epoch << std::endl;
      cdLearner.run();
   }
   
   get_timecourses(&rbm, &data);
   
   //Visualizer samplerviz(20, &data, "RBMsamples");
   
   //rbm.sample(&data, &samplerviz);
  
   
//   terminate(EXIT_SUCCESS);
   
   /*
   Visualizer visualizer(&data); 
   gsl_vector_float *sample = gsl_vector_float_alloc(data.train->size2);
   for (int i = 0; i < 43; ++i){
      gsl_matrix_float_get_row(sample, data.train, i);
      visualizer.add(sample);
   }
   visualizer.plot();
   */
    
   
   return 0; 
}

