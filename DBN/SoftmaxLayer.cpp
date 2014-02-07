//
//  CSoftmaxLayer.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/7/12.
//
//

#include "Layers.h"
#include "DataSets.h"
#include "SupportMath.h"

void SoftmaxLayer::get_expectations(Purpose_t purpose){
   switch (purpose) {
      case LEARNING: m_learning.apply([](float &x){x = expf(x);}); m_learning.norm_rows(); break;
      case TESTING: case GENERATING: case RECOGNITION: m_testing.apply([](float &x){x = expf(x);}); m_testing.norm_rows(); break;
      case VISUALISATION: v_generating.apply([](float &x){x = expf(x);}); v_generating.norm_rows(); break;
   }
}

void SoftmaxLayer::sample() {m_learning.row_pick_top();}

void SoftmaxLayer::get_derivatives() {
   gradient = m_learning;
}

double SoftmaxLayer::reconstructionCost(Matrix &dataMat, Matrix &modelMat){
   reconstruction_cost = cross_validate(dataMat, modelMat);
   reconstruction_cost /= dataMat.dim1;
   return reconstruction_cost;
}

void SoftmaxLayer::shapeInput(DataSet *data){
   Matrix input = data->train;
   float min, max;
   input.min_max(min, max);
   input -= min;
   input /= (max-min);
}

