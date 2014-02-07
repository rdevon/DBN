//
//  ReLULayer.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/6/12.
//
//

#include "Layers.h"
#include "DataSets.h"
#include "SupportMath.h"

void ReLULayer::get_expectations(Purpose_t purpose){
   switch (purpose) {
      case LEARNING: {
         activations = m_learning;
         m_learning.apply([](float &x) {x = softplus(x);});
      } break;
      case TESTING: case GENERATING: case RECOGNITION: {
         m_testing.apply([] (float &x) {x = softplus(x);}); break;
      }
      case VISUALISATION: v_generating.apply([] (float &x) {x = softplus(x);}); break;
   }
}

void ReLULayer::sample() {
   activations.add_relu_noise(m_learning);
   m_learning = activations;
}

void ReLULayer::get_derivatives() {
   gradient = activations;
   //gradient.apply([](float &x){x = (x>0);});
   gradient.apply([] (float &x) {x = sigmoid(x);});
}

//The input needs to be shaped depending on the type of visible layer.
void ReLULayer::shapeInput(DataSet *data){
   Matrix input = data->train;
   float min, max;
   input.min_max(min, max);
   input -= min;
   input /= (max-min);
}
