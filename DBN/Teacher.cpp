//
//  Teacher.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/10/12.
//
//

#include "Teacher.h"
#include "RBM.h"
#include "Viz.h"

ContrastiveDivergence::ContrastiveDivergence(float momentum, int k, int batchsize, int e) : momentum(momentum), k(k), batchsize(batchsize), epochs(e)
{
   monitor = NULL;
   identity = gsl_vector_float_alloc(batchsize);
   gsl_vector_float_set_all(identity, 1);
}

void ContrastiveDivergence::getStats(RBM *rbm){
   
   // Activate the top layer, get the expectations, and sample.
   
   rbm->prop(FORWARD);

   // Positive stats
   rbm->catch_stats(POS);
   
   // Gibbs VH sample k times.
   for(int g = 0; g < k; ++g) rbm->gibbs_VH();

   // Negative stats.
   rbm->catch_stats(NEG);
}

void ContrastiveDivergence::teachRBM(RBM *rbm){
   for (int e = 1; e <= epochs; ++e){
      
      rbm->make_batch(batchsize);
      rbm->sample_flag = SAMPLE;
      
      // Get dimensions
      
      std::cout << std::endl << "Teaching RBM with input, epoch" << e << std::endl << "     K: " << k << std::endl << "     Batch Size: " << batchsize << std::endl;
      
      rbm->init_data();
      // Loop through the input
      int batchnumber = 1;
      
      while (rbm->load_data(TRAIN)){
         
         //monitor->update();
         // Gets statistics by performing CD.
         getStats(rbm);
         
         // Update the parameters.
         rbm->update(this);
         
         // And monitor
         
         if (batchnumber%100 == 0 && monitor != NULL) {
            std::cout << "Batch number: " << batchnumber << std::endl;
            //monitor->update();
         }
         batchnumber+=1;
      }
      rbm->getReconstructionCost();
      monitor->update();
      
   }
   
}
