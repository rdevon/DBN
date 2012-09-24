//
//  Timecourses.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/16/12.
//
//

#include "Timecourses.h"
#include "RBM.h"
#include "Connections.h"
#include "Layers.h"
#include "IO.h"

void get_timecourses(RBM *rbm, Connection *connection, DataSet *data){
   Layer *bot = (Layer*)connection->from;
   Layer *top = (Layer*)connection->to;
   bot->input_edge->train = data->extra;
   rbm->init_data();
   rbm->make_batch((int)data->extra->size1);
   rbm->load_data(TRAIN);
   
   
   rbm->sample_flag = NOSAMPLE;
   rbm->prop(FORWARD);
   /*
   viz.viz = gsl_matrix_float_alloc(top->activations->size1, top->activations->size2);
   gsl_matrix_float_memcpy(viz.viz, top->activations);
   viz.plot();*/
}