//
//  Connections.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/15/12.
//
//

#include "Connections.h"
#include "Layers.h"
#include "Params.h"
#include <math.h>

Connection::Connection(Layer *from_layer, Layer *to_layer)
: from(from_layer), to(to_layer), LearningUnit(to_layer->nodenum, from_layer->nodenum), transmission_scale(1),
weights(to_layer->nodenum, from_layer->nodenum) {
   apply_momentum = true;
   weights.set_gaussian(0.01);
   decay_type = MAXWEIGHT;
}

Transmit_t Connection::transmit_signal(Direction_t direction, Purpose_t purpose, Sample_flag_t s_flag){
   CBLAS_TRANSPOSE transFlag;
   Layer *input, *output;
   float trans_scale;
   
   if (direction == UP) {
      input = from;
      output = to;
      transFlag = CblasTrans;
      trans_scale = transmission_scale;
      if ((purpose == TESTING || purpose == RECOGNITION) && from->noisy) trans_scale*=(1-from->noise);
   }
   
   else if (direction == DOWN) {
      input = to;
      output = from;
      transFlag = CblasNoTrans;
      trans_scale = 1/transmission_scale;
      if ((purpose == TESTING || purpose == GENERATING) && to->noisy) trans_scale*=(1-to->noise);
   }
   
   switch (purpose) {
      case LEARNING:      output->m_learning.times_plus(   input->m_learning, weights,   trans_scale, output->reset_switch, CblasNoTrans, transFlag); break;
      case TESTING:
      case GENERATING:
      case RECOGNITION:   output->m_testing.times_plus(    input->m_testing, weights,    trans_scale, output->reset_switch, CblasNoTrans, transFlag); break;
      case VISUALISATION: output->v_generating.times_plus( input->v_generating, weights, trans_scale, output->reset_switch, CblasNoTrans, transFlag); break;
   }
   output->reset_switch = 1;
   return TRANSMIT_SUCCESS;
}

void Connection::transmit_data(const Matrix &from, Matrix &to) {
   to.times_plus(from, weights, 1, 1, CblasNoTrans, CblasTrans);
}

void Connection::catch_stats(Stat_flag_t stat_flag){
   int batchsize = (int)to->m_learning.dim1;
   switch (stat_flag) {
      case POS: gradient.times_plus(to->m_learning, from->m_learning, learning_rate/batchsize, 0, CblasTrans, CblasNoTrans); break;
      case NEG: gradient.times_plus(to->m_learning, from->m_learning, -learning_rate/batchsize, 1, CblasTrans, CblasNoTrans); break;
   }
}

void Connection::update(Teacher& teacher, bool apply_gain) {
   param = &weights;
   weight_max_length = to->max_input;
   LearningUnit::update(teacher, apply_gain);
   //if (from->type == SOFTMAX)
   //   std::cout << weights;
}