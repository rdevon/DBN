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
#include "SupportFunctions.h"
#include "Layers.h"
#include "Connections.h"
#include "Monitors.h"
#include <math.h>
#include "Monitor_Units.h"

//------ For increasing and decreasing the learning rate during visualization.

void Teacher::multiply_rate() {learning_multiplier *=2;}
void Teacher::divide_rate() {learning_multiplier /=2;}

void LearningUnit::update(Teacher &teacher, bool apply_gain) {
   if (teacher.learning_multiplier >1) gradient*=teacher.learning_multiplier;
   
   if (gain) gradient *= learning_gain;
   if (gain && apply_gain) {
      Matrix *prev;
      if (teacher.momentum > 0) prev = &prev_velocity;
      else                      prev = &prev_gradient;
      learning_gain.adjust_gain(*prev, acc_gradient);
      prev_gradient = acc_gradient;
      prev_velocity = acc_velocity;
      acc_gradient.set_all(0);
      acc_velocity.set_all(0);
   }
   if (teacher.momentum > 0) {
      *param += velocity;
      velocity = (teacher.momentum*velocity) + gradient;
   }
   else *param += gradient;

   if (gain) {
      acc_gradient += gradient;
      acc_velocity += velocity;
   }
   
   switch (decay_type) {
      case NONE: break;
      case MAXWEIGHT: {
         (*param).flatten_rows(weight_max_length);
         //velocity.flatten_rows(2*weight_max_length);
      }
      default: {
         float d_rate;
         if (decay_rate == AUTO) d_rate = learning_rate/param->dim2*0.0002;
         *param -= d_rate*(*param)*learning_gain;
      } break;
   }
}


