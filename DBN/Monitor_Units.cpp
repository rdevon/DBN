
//
//  File.cpp
//  DBN
//
//  Created by Devon Hjelm on 10/23/12.
//
//

#include "Monitor_Units.h"
#include "MLP.h"
#include "Layers.h"
#include "Monitors.h"
#include "DataSets.h"
#include "Connections.h"
#include "SupportMath.h"

Layer_to_Data_Monitor::Layer_to_Data_Monitor(MLP *mlp, Layer *feature_layer) :
mlp(mlp), feature_layer(feature_layer) {}

void Layer_to_Data_Monitor::update() {Grid_Viz_Unit::update();}

Feature_to_Data_M_Unit::Feature_to_Data_M_Unit (int f, MLP *mlp, Layer *feature_layer) :
Stacked_Tex_Unit(mlp->viz_layer->data->dims[0], mlp->viz_layer->data->dims[1]), mlp(mlp), feature(f), feature_layer(feature_layer) {
   color[0] = 1, color[1] = 0, color[2] = 0, color [3] = 1;
}

void Feature_to_Data_M_Unit::update() {
   if (!on) {viz_matrix.set_all(WHITE);}
   
   feature_layer->set_component(feature);
   mlp->transmit(DOWN, GENERATING, NOSAMPLE);
   
   mlp->viz_layer->v_generating.min_max(_min, _max);
   mlp->viz_layer->data->apply_mask(mlp->viz_layer->data->image, mlp->viz_layer->v_generating);
   viz_matrix.load(mlp->viz_layer->data->image, level);
}

#if 0

Timecourse_M_Unit::Timecourse_M_Unit (int feat, MLP *source_mlp, Layer *f_layer) : feature(feat), feature_layer(f_layer) {
   on = true;
   color[0] = 0, color[1] = 1, color[2] = 0, color [3] = 1;
   mlp = source_mlp;
   line_set = gsl_vector_float_calloc(220);
}

void Timecourse_M_Unit::update() {
   
   Layer *to = feature_layer;
   mlp->init_data();
   mlp->transmit(GENERATING, FORWARD, NOSAMPLE);
   line_set =
   gsl_vector_float_view timecourse = gsl_matrix_float_row(to->expectations, feature);
   gsl_vector_float_memcpy(line_set, &timecourse.vector);
}

#endif

Reconstruction_Cost_Monitor::Reconstruction_Cost_Monitor(MLP *monitored_mlp, int epochs) :
status(OK), mlp(monitored_mlp), Plot_Unit(epochs), epoch(0) {}

void Reconstruction_Cost_Monitor::update() {
   mlp->get_reconstruction_cost();
   
   if (epoch >= line_set.dim2) {
      Vector newset(2*line_set.dim2);
      newset.fill_submatrix(line_set, 0, 0);
      line_set = newset;
   }
   line_set(0,epoch) = mlp->reconstruction_cost;
   ++epoch;
}

void Reconstruction_Cost_Monitor::check() {
   if (epoch > 1 && line_set(0,epoch-1) > 2*(line_set(0, epoch-2))) {
      status = BROKEN;
      return;
   }
   
   float rc = mlp->reconstruction_cost;
   int count = 0;
   float mean_neg_gradient = 0;
   float previous = 0;
   int e = epoch - 1;
   int i = std::max(0,e-15);
   for (; i <= e; ++i) {
      float val = line_set(0,i);
      
      if (previous != 0) {
         mean_neg_gradient += (previous-val);
         count += 1;
      }
      previous = val;
   }
   mean_neg_gradient /= count;
   mean_neg_gradient/=rc;
   float gradient;
   if (e > 0) {
      float minus = line_set(0,e);
      float plus = line_set(0,e-1);
      gradient = fabsf(plus - minus);
   }
   else gradient = rc;
   
   gradient/=rc;
   //std::cout << "Mean reconstruction cost gradient: " << gradient << ", " << mean_neg_gradient << " mean neg grad." << std::endl;
   if (gradient < 0.000001)
      status = DONE;
   else if (gradient < 0.005) status = SLOW;
   else status = OK;
}