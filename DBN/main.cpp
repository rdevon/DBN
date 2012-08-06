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

int main (int argc, const char * argv[])
{
   //--------------RNG INIT STUFF
   long seed;
   r = gsl_rng_alloc (gsl_rng_rand48);     // pick random number generator
   seed = time (NULL) * getpid();
   gsl_rng_set (r, seed);                  // set seed
   //--------------
   //LOAD DATASET and INIT
   
   DataSet data;
   data.loadfMRI();
   
   Visualizer viz(&data);
   
   //---------DONE INIT
  
   //INIT RBM
   
   GaussianLayer baselayer((int)data.train->size2);
   
   baselayer.addLayer(40);
   
   RBM rbm(&baselayer);
   
   //--------------
   float learningrate = 0.001;
   float weightcost = 0.0001;
   float hiddenlayernodes = 50;
   float sparsitytarget = 0.2;
   float decayrate = 0.9;
   float sparsitycost = .01;
   float batchsize = 1;
   
   CD cdLearner(learningrate, weightcost, hiddenlayernodes, sparsitytarget, decayrate, sparsitycost, &viz, batchsize);
   
   rbm.bot->shapeInput(data.train);
   //LEARNING!!!!!!!!!
   for (int epoch = 0; epoch < 100 ; ++epoch){
      std::cout << std::endl << "Epoch " << epoch << std::endl;
      cdLearner.teach(&rbm, data.train);
      std::cout << "Reconstruction error: " << rbm.reconstructionError_ << std::endl;
   }
   
   Visualizer samplerviz(&data, "RBMsamples");
   
   rbm.sample(&data, &samplerviz);
  
   
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

