//
//  Timecourses.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/16/12.
//
//

#include "Timecourses.h"

void get_timecourses(RBM *rbm, DataSet *data){
   Input_t *course = data->extra;
   Layer *bot = rbm->c1_->bot_;
   Layer *top = rbm->c1_->top_;
   
   rbm->makeBatch((int)course->size1);
   
   Visualizer viz(top->nodenum_, data, "fMRI_timecourses");
   Visualizer viz2(top->nodenum_, data, "fMRI_timecourses2");
   
   gsl_matrix_float_transpose_memcpy(bot->expectations_, course);
   rbm->up_act_->s_flag_ = NOSAMPLE;
   rbm->up_act_->activate();
   top->getExpectations();
   
   viz.viz = gsl_matrix_float_alloc(top->activations_->size1, top->activations_->size2);
   viz2.viz = gsl_matrix_float_alloc(top->activations_->size1, top->activations_->size2);
   gsl_matrix_float_memcpy(viz.viz, top->activations_);
   gsl_matrix_float_memcpy(viz2.viz, top->expectations_);
   viz.plot();
   viz2.plot();
}