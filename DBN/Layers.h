//
//  Layers.h
//  DBN
//
//  Created by Devon Hjelm on 7/10/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_Layers_h
#define DBN_Layers_h

#include <iostream>
#include "Types.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include "SupportMath.h"
#include "SupportFunctions.h"
#include "IO.h"
#include "Teacher.h"
#include "Layers.h"

class Layer;

/////////////////////////////////////
// Layer class
/////////////////////////////////////

class Layer : public LearningUnit {
public:
   bool frozen;
   bool expectation_up_to_date, sample_up_to_date, learning_up_to_date;
   
   int nodenum_, batchsize_;
   
   //--------All of the activations are done as nxb matrices, where n is the number of nodes and b is the batch size
   gsl_matrix_float *activations_;                 // The literal unit activations.
   gsl_matrix_float *samples_;                     // These are for the binary units specifically, but might be needed for others.  The sigmoid function is
   gsl_matrix_float *expectations_;                // The statistical mean of the samples.  These are good when doing less noisy analysis (see activation flags)
   gsl_vector_float *m_factor_;                    // A multiplicative factor for signals (this is for gaussian layers especially)
   
   gsl_vector_float *biases_;                      // Biases
   gsl_matrix_float *batchbiases_;                 // For batch processing
   
   float energy_;                                  // Energy of the layer *TODO*
   
   ~Layer(){};
   Layer(int n);                                   // Constructor for the Layer
   
   // Unit Functions------------
   virtual void sample() = 0;         // Begin sampling.  If sample flag is on, calculate the samples, set samples to the expectation.
   virtual void getExpectations() = 0;             // Find the expectated values for the layer
   
   
   // Structure Functions------------
   void clear();
   virtual void makeBatch(int batchsize);          // Changes all of the unit matrices into matrices of size
                                                   // nodenum_ x batchsize_
   void expandBiases();                            // The biases are vectors, but it's nice to have matrix versions as well.
   virtual void shapeInput(DataSet* data) = 0;    // Depending on the type of layer you need to shape the input.  Should be useful in DBNS as well.
   
   // Energy Functions-------------
   virtual float reconstructionCost(gsl_matrix_float *dataMat, gsl_matrix_float *modelMat) = 0;
   virtual void getEnergy() = 0;
   virtual float freeEnergy_contibution() = 0;
   
   // Update Functions-------------
   void catch_stats(Stat_flag_t, Sample_flag_t);
   virtual void update(ContrastiveDivergence*);
};

/////////////////////////////////////
// Sigmoid layer class
/////////////////////////////////////

class SigmoidLayer : public Layer {
public:
   
   SigmoidLayer(int n) : Layer(n){
      biases_ = gsl_vector_float_alloc(nodenum_);
      gsl_vector_float_set_all(biases_, 0); // This is to force sparsity in simple cases.  Set to some negative number.  Good for analysis
   }
   
   void sample();
   void getExpectations();
   void shapeInput(DataSet *data);
   
   float reconstructionCost(gsl_matrix_float *dataMat, gsl_matrix_float *modelMat);
   void getEnergy(){}
   float freeEnergy_contibution();
   
   void update(ContrastiveDivergence*);
   
};
/////////////////////////////////////
// Rectified Linear Unit layer class *TODO*
/////////////////////////////////////

class ReLULayer : public Layer {
public:
   ReLULayer(int n) : Layer(n){
      biases_ = gsl_vector_float_calloc(nodenum_);
   }
   
   void sample();
   void getExpectations();
   void shapeInput(DataSet* data);
   
   float reconstructionCost(gsl_matrix_float *dataMat, gsl_matrix_float *modelMat);
   void getEnergy(){}
   float freeEnergy_contibution();
   
   void update(ContrastiveDivergence*);
};

/////////////////////////////////////
// Continuous Softmax layer class
/////////////////////////////////////

class CSoftmaxLayer : public Layer {
public:
   CSoftmaxLayer(int n) : Layer(n){
      biases_ = gsl_vector_float_calloc(nodenum_); //Maybe .5?
   }
   
   void sample();
   void getExpectations();
   void shapeInput(DataSet *data);
   
   float reconstructionCost(gsl_matrix_float *dataMat, gsl_matrix_float *modelMat){return 0;}
   void getEnergy(){}
   float freeEnergy_contibution();
   
   void update(ContrastiveDivergence*){}
};

/////////////////////////////////////
// Gaussian layer class
/////////////////////////////////////

class GaussianLayer : public Layer {
public:
   
   float setsigma;
   
   GaussianLayer(int n);
   
   gsl_vector_float *quad_coefficients;
   gsl_vector_float *sigmas;
   
   void sample();
   void getExpectations();
   void getSigmas();
   void shapeInput(DataSet *data);
   
   void makeBatch(int batchsize);
   
   float reconstructionCost(gsl_matrix_float *dataMat, gsl_matrix_float *modelMat);
   void getEnergy(){}
   float freeEnergy_contibution(){ return 0;}
   
   void update(ContrastiveDivergence*);
};

#endif
