//
//  Layers.h
//  DBN
//
//  Created by Devon Hjelm on 7/10/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_Layers_h
#define DBN_Layers_h

#include "Teacher.h"

#include "Params.h"
#include "Matrix.h"

/////////////////////////////////////
// Layer class
/////////////////////////////////////

class DataSet;

class Layer : public LearningUnit {
public:
   
   // MEMBERS -------------------------------------------------------------------------------------
   
   Layer_t              type;
   float                max_input;
   size_t               nodenum;
   float                noise;
   bool                 noisy;
   int                  reset_switch;
   DataSet              *data;
   
   //--------Vectors and matrices
   
   Vector               v_generating;                 // Vector for generation
   Matrix               m_learning;                   // Matrix for learning (generally batch)
   Matrix               m_testing;                    // Matrix for error testing (batch)
   Matrix               m_dropout;                    // Matrix for implimenting dropout;
   
   Vector               biases;                       // Biases
   Vector               rec_biases;
   
   //Matrix     extra;
   //Vector     sample_vector;
   
   float                energy;                       // Energy of the layer *TODO*
   double               reconstruction_cost;
   
   // CONSTRUCTORS ---------------------------------------------------------------------------------
   
   Layer(size_t nn, size_t bs, size_t ts);                                   // Constructor for the Layer
   Layer(const Layer&);
   virtual Layer *clone() = 0;
   void reset(){
      biases.set_all(0);
   }
   virtual void set_defaults() = 0;
   virtual void set_with_fanin(Layer& other) = 0;
   
   // Unit Functions------------
   
   int pull_data(Purpose_t purpose);
   void finish_activation(Purpose_t, Sample_flag_t);
   void make_noise();
   void apply_noise();
   virtual void get_expectations(Purpose_t purpose) = 0;
   virtual void sample() = 0;
   virtual void get_derivatives() = 0;
   
   // Structure Functions------------

   virtual void shapeInput(DataSet* data) = 0;    // Depending on the type of layer you need to shape the input.  Should be useful in DBNS as well.
   
   // Energy Functions-------------
   virtual double reconstructionCost(Matrix &dataMat, Matrix &modelMat);
   virtual void getEnergy(){};
   virtual float freeEnergy_contibution(){return 0;}
   
   // Update Functions-------------
   void catch_stats(Stat_flag_t);
   void update(Teacher&, bool);
   
   // For Label and Stimulus Analysis.  Different types of units and data have different notions of orthogonal, and we need to capture that here.
   virtual void set_component(int i);
   
   void save();
};

/////////////////////////////////////
// Sigmoid layer class
/////////////////////////////////////

class SigmoidLayer : public virtual Layer {
public:
   SigmoidLayer(size_t nn, size_t bs, size_t ts) : Layer(nn,bs,ts){ type = SIGMOID; set_defaults();}
   SigmoidLayer(const SigmoidLayer& other) : Layer(other) {}
   SigmoidLayer *clone() {return new SigmoidLayer(*this);}
   void set_defaults();
   
   void get_expectations(Purpose_t purpose);
   void sample();
   void shapeInput(DataSet *data);
   void get_derivatives();
   void set_with_fanin(Layer& other);
   
   double reconstructionCost(Matrix &dataMat, Matrix &modelMat);
};

/////////////////////////////////////
// Rectified Linear Unit layer class *TODO*
/////////////////////////////////////

class ReLULayer : public Layer {
public:
   ReLULayer(size_t nn, size_t bs, size_t ts);
   ReLULayer(const ReLULayer& other) : Layer(other), activations(other.activations) {}
   ReLULayer *clone() {return new ReLULayer(*this);}
   void set_defaults();
   void set_with_fanin(Layer& other);
   
   Matrix activations;
   
   void get_expectations(Purpose_t purpose);
   void sample();
   void shapeInput(DataSet* data);
   void get_derivatives();
};

/////////////////////////////////////
// Gaussian layer class
/////////////////////////////////////

class GaussianLayer : public Layer {
public:
   GaussianLayer(size_t nn, size_t bs, size_t ts);
   GaussianLayer(const GaussianLayer& other) : Layer(other){}
   GaussianLayer *clone() {return new GaussianLayer(*this);}
   void set_defaults();
   void set_with_fanin(Layer& other);
   
   void get_expectations(Purpose_t purpose);
   void sample();
   void shapeInput(DataSet *data);
   void get_derivatives();
};

/////////////////////////////////////
// Softmax layer class
/////////////////////////////////////

class SoftmaxLayer : public Layer {
public:
   SoftmaxLayer(size_t nn, size_t bs, size_t ts);
   SoftmaxLayer(const SoftmaxLayer& other) : Layer(other){}
   SoftmaxLayer *clone() {return new SoftmaxLayer(*this);}
   void set_defaults();
   void set_with_fanin(Layer& other);
   
   void get_expectations(Purpose_t purpose);
   void sample();
   void shapeInput(DataSet *data);
   void get_derivatives();
   
   double reconstructionCost(Matrix &dataMat, Matrix &modelMat);
   void finish_activation(Sample_flag_t);
};

std::ostream& operator<<(std::ostream& out,Layer& layer);

#endif
