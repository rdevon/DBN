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
   data.loadMNIST();
   
   Visualizer viz(&data);
   
   
   //---------DONE INIT
  
   //INIT RBM
   SigmoidLayer baselayer(data.height*data.width);
   
   baselayer.addLayer(500);
   
   RBM rbm(&baselayer);
   
   //--------------
   CD cdLearner(.1, 15, &viz,20);
   
   //LEARNING!!!!!!!!!
   for (int epoch = 0; epoch < 20 ; ++epoch){
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

