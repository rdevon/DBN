//
//  Connections.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/15/12.
//
//

#include "Connections.h"
#include "Layers.h"
#include "Types.h"

Connection::Connection(Layer *from_layer, Layer *to_layer) {
   
   level = 0;
   
   from = from_layer;
   to = to_layer;
   
   from->forward_edges.push_back(this);
   to->backward_edges.push_back(this);
   
   weights = gsl_matrix_float_alloc(to_layer->nodenum, from_layer->nodenum);
   for (int i = 0; i < to_layer->nodenum; ++i)
      for (int j = 0; j < from_layer->nodenum; ++j)
         gsl_matrix_float_set(weights, i, j, (float)gsl_ran_gaussian(r, 0.01));
   
   mat_update = gsl_matrix_float_calloc(to_layer->nodenum, from_layer->nodenum);
   node_projections = gsl_vector_float_alloc(from_layer->nodenum);
}

void Connection::make_batch(int batchsize){
   Layer* from_layer = (Layer*)from;
   Layer* to_layer = (Layer*)to;
   if (batchsize == 0) batchsize = (int)from_layer->input_edge->train->size1;
   from_layer->make_batch(batchsize);
   to_layer->make_batch(batchsize);
}

void Connection::init_activation(MLP *mlp){
   Layer *layer;
   if (direction_flag == FORWARD) layer = (Layer*)to;
   else if (direction_flag == BACKWARD) layer = (Layer*)from;
   layer->init_activation(mlp);
}

int Connection::transmit_signal(Sample_flag_t s_flag){
   Layer* from_layer = (Layer*)from;
   Layer* to_layer = (Layer*)to;
   CBLAS_TRANSPOSE_t transFlag;
   Layer *input_layer, *output_layer;
   
   if (direction_flag == FORWARD) {
      input_layer = from_layer;
      output_layer = to_layer;
      transFlag = CblasNoTrans;
   }
   
   else if (direction_flag == BACKWARD){
      input_layer = to_layer;
      output_layer = from_layer;
      transFlag = CblasTrans;
   }
   
   switch (output_layer->status) {
      case FROZEN    : return 1;
      case WAITING   : output_layer->visits_waiting-=1; break;
      case READY     : std::cout << "shouldn't get here 1 "<< output_layer->status << " " << READY << std::endl; break;
      case DONE      : return 1;
      case OFF       : return 1;
   }
   
   
   switch (input_layer->status) {
      case WAITING   : return 0;
      case READY     : std::cout << "shouldn't get here 2 " << input_layer->status << std::endl;
      case OFF       : return 1;
      default        :
      {
         if (input_layer->noisy && s_flag == SAMPLE) input_layer->apply_noise();
         
         gsl_blas_sgemm(transFlag, CblasNoTrans, 1, weights, input_layer->samples, 1, output_layer->activations);
         if (output_layer->visits_waiting == 0) output_layer->finish_activation(s_flag);
         return 1;
      }
   }
}

void Connection::catch_stats(Stat_flag_t stat_flag, Sample_flag_t sample_flag){
   Layer *from_layer = (Layer*)from;
   Layer *to_layer = (Layer*)to;
   
   stat1 = to_layer->stat1;
   stat2 = from_layer->stat1;
   stat3 = to_layer->stat2;
   stat4 = from_layer->stat2;
   
   from_layer->catch_stats(stat_flag, sample_flag);
   if      (stat_flag == NEG) to_layer->catch_stats(stat_flag, NOSAMPLE);
   else if (stat_flag == POS) to_layer->catch_stats(stat_flag, SAMPLE);
}

void Connection::update(ContrastiveDivergence* teacher){
   Layer *from_layer = (Layer*)from;
   Layer *to_layer = (Layer*)to;
   
   from_layer->learning_rate = learning_rate;
   to_layer->learning_rate = learning_rate;
   from_layer->update(teacher);
   to_layer->update(teacher);
   
   gsl_matrix_float *weight_update = mat_update;
   float rate = learning_rate/((float)teacher->batchsize);
   //learning_rate/=(float)teacher->batchsize_;
   gsl_blas_sgemm(CblasNoTrans, CblasTrans , rate, stat1, stat2, teacher->momentum, weight_update);
   gsl_blas_sgemm(CblasNoTrans, CblasTrans , -rate, stat3, stat4, 1, weight_update);
   
   gsl_matrix_float *weightdecay = gsl_matrix_float_alloc(weights->size1, weights->size2);
   gsl_matrix_float_memcpy(weightdecay, weights);
   gsl_matrix_float_scale(weightdecay, decay);
   gsl_matrix_float_sub(weight_update, weightdecay);
   
   gsl_matrix_float_add(weights, weight_update);
   
   gsl_matrix_float_free(weightdecay);
}

void Connection::init_data(){
   Layer *from_layer = (Layer*)from;
   from_layer->input_edge->index = 0;
}

int Connection::load_data(Data_flag_t data_flag){
   Layer *from_layer = (Layer*)from;
   if (from_layer->status == DONE) return 1;
   return from_layer->load_data(data_flag);
}


