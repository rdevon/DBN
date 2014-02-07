//
//  Layers.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/18/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "Layers.h"
#include "DataSets.h"
#include "SupportMath.h"
#include "SupportFunctions.h"

Layer::Layer(size_t nn, size_t bs, size_t ts)
: LearningUnit(1, nn), nodenum(nn), noise(0), data(NULL),
v_generating(nn), m_learning(bs,nn), m_testing(ts,nn), biases(nn), rec_biases(nn), m_dropout(bs,nn),
energy(0)
{
   noisy = false;
   decay_type = NONE;
}

Layer::Layer(const Layer &other) : LearningUnit(other), nodenum(other.nodenum), noise(other.noise), data(other.data),
noisy(other.noisy), v_generating(other.v_generating), m_learning(other.m_learning), m_testing(other.m_testing), biases(other.biases), rec_biases(other.rec_biases),
m_dropout(other.m_dropout) ,type(other.type), max_input(other.max_input) {}

void SigmoidLayer::set_defaults() {
   noise = 0.5;
   max_input = 10;
   learning_rate = 1;
}

void SigmoidLayer::set_with_fanin(Layer &other) {
}

ReLULayer::ReLULayer(size_t nn, size_t bs, size_t ts) : Layer(nn,bs,ts), activations(bs,nn) {
   type = RELU;
   set_defaults();
}

void ReLULayer::set_defaults() {
   noise = 0.5;
   max_input = 1;
   learning_rate = .000002; //0.000002
}

void ReLULayer::set_with_fanin(Layer &other) {
   weight_max_length = other.nodenum;
}

GaussianLayer::GaussianLayer(size_t nn, size_t bs, size_t ts) : Layer(nn,bs,ts) {
   type = GAUSSIAN;
   set_defaults();
   biases.set_gaussian(0.01);
}

void GaussianLayer::set_with_fanin(Layer &other) {
}

void GaussianLayer::set_defaults() {
   max_input = 10;
   noise = 0;
   learning_rate = 1;
}

SoftmaxLayer::SoftmaxLayer(size_t nn, size_t bs, size_t ts) : Layer(nn,bs,ts){
   type = SOFTMAX;
   set_defaults();
}

void SoftmaxLayer::set_defaults() {
   noise = 0;
   max_input = 1;
   learning_rate = 0;
}

void SoftmaxLayer::set_with_fanin(Layer &other) {
}

int Layer::pull_data(Purpose_t purpose){
   if (!data) error("No data to pull");
   
   Matrix *input;
   Matrix *output;
   
   switch (purpose) {
      case LEARNING: {
         input = &data->train;
         output = &m_learning;
      } break;
      case TESTING: case GENERATING: case RECOGNITION: {
         input = &data->validation;
         output = &m_testing;
      } break;
      case VISUALISATION: {
         input = &data->data;
         output = &v_generating;
      } break;
   }
   
   if (data->index + output->dim1 > input->dim1) return 0;
   output->copy_submatrices(*input, data->index);
   data->index += output->dim1;
   if (data->index >= input->dim1) return 0;
   return 1;
}

void Layer::make_noise() {
   if (!noisy) return;
   m_dropout.set_all(1);
   m_dropout.dropout(noise);
}

void Layer::apply_noise(){
   if (!noisy) return;
   m_learning*=m_dropout;
}

void Layer::finish_activation(Purpose_t purpose, Sample_flag_t s_flag){
   
   switch (purpose) {
      case LEARNING: {
         m_learning += biases;
         get_expectations(purpose);
         if (s_flag == SAMPLE) sample();
         if (noisy) apply_noise();
      } break;
      case TESTING: case GENERATING: {
         m_testing += biases;
         get_expectations(purpose);
         if (s_flag == SAMPLE) sample();
      } break;
      case RECOGNITION: {
         m_testing += rec_biases;
         get_expectations(purpose);
         if (s_flag == SAMPLE) sample();
      }
      case VISUALISATION: {
        // v_generating += biases;
         get_expectations(purpose);
      } break;
   }
}

void Layer::update(Teacher &teacher, bool apply_gain){
   param = &biases;
   if (noisy) { gradient*=m_dropout.mean_image();}
   LearningUnit::update(teacher, false);
}

void Layer::catch_stats(Stat_flag_t stat){
   Vector mi = m_learning.mean_image();
   switch (stat) {
      case POS: gradient = learning_rate*mi; break;
      case NEG: gradient -= learning_rate*mi; break;
   }
}

double Layer::reconstructionCost(Matrix &dataMat, Matrix &modelMat){
   float rc=0;
   dataMat -= modelMat;
   dataMat.apply([&rc](float x) {rc += pow(x,2);});
   reconstruction_cost = rc/(dataMat.dim1);
   return reconstruction_cost;
}

void Layer::set_component(int i) {
   v_generating.set_all(0);
   v_generating(0,i) = 1;
}


