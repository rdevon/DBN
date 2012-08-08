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
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_rng.h"
#include <gsl/gsl_randist.h>
#include "SupportMath.h"
#include "SupportFunctions.h"
#include "IO.h"
#include <gsl/gsl_statistics.h>

class Layer;
class SigmoidLayer;
class GaussianLayer;
class ReLULayer;
class CSoftmaxLayer;

/////////////////////////////////////
// Activator class
/////////////////////////////////////

// The activator class.  This is for activating a unit given another unit.
// It might seem strange to treat these as separate objects, but there are some options that are nice to have when you're activating units
// (independent of type).  For instance, you might want to use the activations of a unit when finding the activations and probs, but other times you might want
// less noise and to use the probabilities themselves.  Just set the flag depending on the type of activations.
// In addition, I think the acticator visitor will become more useful as a layer has multiple acitvation sources.
class PreActivator{
public:
   Layer *sL1_, *sL2_, *sL3_;
   
   Up_flag_t up_;
   
   ~PreActivator(){}
   PreActivator(Up_flag_t up, Layer *sL1, Layer *sL2 = NULL, Layer *sL3 = NULL) : up_(up), sL1_(sL1), sL2_(sL2), sL3_(sL3) {}
   
   void preactivate(Layer*);
};

/////////////////////////////////////
// Layer class
/////////////////////////////////////

class Layer {
public:
   bool frozen, on;
   PreActivator *preactivator;
   
   int nodenum_, batchsize_;
   
   //All of the activations are done as nxb matrices, where n is the number of nodes and b is the batch size
   gsl_matrix_float *probabilities_;            // The statistical mean of the activations.  These are good when doing less noisy analysis (see activation flags)
   gsl_matrix_float *activations_;      // The literal unit activations.
   gsl_matrix_float *preactivations_;   // These are for the binary units specifically, but might be needed for others.  The sigmoid function is
                                       // problematic when concluting the reconstruction error.
   
   gsl_vector_float *biases_;           // Biases
   gsl_matrix_float *batchbiases_;      // For batch processing
   gsl_matrix_float *weights_;          // Weights to layer above
   gsl_matrix_float *weightsT_;         // Weights to layer above transport *TODO* These are for the eventual autoencoder... possibly
   
   Layer *up;                          // The layer above
   
   float freeEnergy_;                   // Free energy of layer
   
   Layer(int n);
   
   void clear();
   void initializeWeights();
   void addLayer(int newnodes);
   void addLayer(Layer *layer);
   Layer *getTop();
   void makeBatch(int batchsize);
   
   void activate(Activation_flag_t act);
   
   virtual void getFreeEnergy() = 0;
   virtual void setProbs() = 0;
   virtual void shapeInput(Input_t* input) = 0;                         // Depending on the type of layer you need to shape the input.  Should be useful in DBNS as well.
   virtual double reop(double arg) = 0;
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
   
   void getFreeEnergy();
   void setProbs();
   
   void shapeInput(Input_t* input);
   double reop(double arg){
      return softplus(arg);}
};
/////////////////////////////////////
// Rectified Linear Unit layer class *TODO*
/////////////////////////////////////

class ReLULayer : public Layer {
public:
   ReLULayer(int n) : Layer(n){
      biases_ = gsl_vector_float_calloc(nodenum_);
   }
   
   void getFreeEnergy(); //*TODO*
   void setProbs();
   void shapeInput(Input_t* input);
   
   double reop(double arg){
      return arg;}
};

/////////////////////////////////////
// Continuous Softmax layer class
/////////////////////////////////////

class CSoftmaxLayer : public Layer {
public:
   CSoftmaxLayer(int n) : Layer(n){
      biases_ = gsl_vector_float_calloc(nodenum_); //Maybe .5?
   }
   
   void getFreeEnergy(); //*TODO*
   void setProbs();
   void shapeInput(Input_t* input);
   
   double reop(double arg){
      return arg;}
};

/////////////////////////////////////
// Gaussian layer class
/////////////////////////////////////

class GaussianLayer : public Layer {
public:
   
   GaussianLayer(int n) : Layer(n) {
      biases_ = gsl_vector_float_calloc(nodenum_);
      for (int i = 0; i < nodenum_; ++i)
         gsl_vector_float_set(biases_, i, (float)gsl_ran_gaussian(r, 0.01));
   }
   
   void getFreeEnergy(){} //*TODO*
   void setProbs();
   void shapeInput(Input_t* input);
   double reop(double arg){
      return arg;}
};

#endif
