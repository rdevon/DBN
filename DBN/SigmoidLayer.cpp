//
//  SigmoidLayer.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/18/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include "Layers.h"
#include "DataSets.h"
#include "SupportMath.h"

void SigmoidLayer::get_expectations(Purpose_t purpose) {
   switch (purpose) {
      case LEARNING: m_learning.apply([](float &x){x = sigmoid(x);}); break;
      case TESTING: case GENERATING: case RECOGNITION: m_testing.apply([](float &x){x = sigmoid(x);}); break;
      case VISUALISATION: v_generating.apply([](float &x){x = sigmoid(x);}); break;
   }
}

void SigmoidLayer::sample() {m_learning.sample();}

void SigmoidLayer::get_derivatives() {
   gradient = m_learning;
   gradient.apply([] (float &x) {x = x*(1-x);});
}

double SigmoidLayer::reconstructionCost(Matrix &dataMat, Matrix &modelMat){
   reconstruction_cost = cross_validate(dataMat, modelMat);
   reconstruction_cost /= dataMat.dim1;
   return reconstruction_cost;
}

void SigmoidLayer::shapeInput(DataSet *data){
   Matrix input = data->train;
   float min, max;
   input.min_max(min, max);
   input -= min;
   input /= (max-min);
}