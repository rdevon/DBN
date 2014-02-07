//
//  Monitors.cpp
//  DBN
//
//  Created by Devon Hjelm on 10/23/12.
//
//
#include "Monitors.h"
#include "Viz.h"
#include "Monitor_Units.h"
#include "Teacher.h"
#include "DBN.h"
#include "Layers.h"
#include "DataSets.h"
#include "MLP.h"
#include "IO.h"

Monitor::Monitor(){
#ifdef USEGL
   viz = new Visualizer(1000,1000);
   the_viz = viz;
#endif
   the_monitor = this;
   threshold = 0;
}

Monitor::Monitor(Visualizer *old_viz) {
#ifdef USEGL
   viz = old_viz;
   viz->clear();
   the_viz = viz;
#else
   viz = NULL;
   teacher = NULL;
#endif
   the_monitor = this;
   threshold = 0;
}

void Monitor::load_units_into_visualizer(Visualizer *viz) {
   for (auto unit:grid_units) unit->load_into_visualizer(viz);
   for (auto unit:stat_units) unit->load_into_visualizer(viz);
}

void Monitor::update() {
   do {
      for (auto unit:grid_units) unit->update();
#ifdef USEGL
      viz->update(this);
   } while (the_viz->pause);
#else
   } while (false);
#endif
   recvd_press = false;
}

void Monitor::peek() {
   if (recvd_press) update();
}

void Monitor::update_stats() {
   for (auto unit:stat_units) unit->update();
}

void Monitor::send_stop_signal() {
   if (teacher == NULL) exit(EXIT_SUCCESS);
   teacher->learning = false;
}

void Monitor::move_down_stack() {
   for (auto unit:stacked_units) unit->step_back();
}

void Monitor::move_up_stack() {
   for (auto unit:stacked_units) unit->step_forward();
}

void Monitor::view() {
   while (1) this->update();
}

//----------------------------------------

Layer_to_Data_Monitor *Monitor::construct_LTDM(MLP *mlp, Layer *layer) {
   Layer_to_Data_Monitor *layer_monitor = new Layer_to_Data_Monitor(mlp, layer);
   for (int i = 0; i < layer->nodenum; ++i) {
      Feature_to_Data_M_Unit *feature_stack = new Feature_to_Data_M_Unit(i, mlp, layer);
      feature_stack->set_size(mlp->viz_layer->data->dims[0], mlp->viz_layer->data->dims[1],0);
      feature_stack->level = mlp->viz_layer->data->dims[2]/2;
      feature_stack->stack_bot = 0;
      feature_stack->stack_top = (int)mlp->viz_layer->data->dims[2]-1;
      layer_monitor->units.push_back(feature_stack);
      stacked_units.push_back(feature_stack);
   }
   grid_units.push_back(layer_monitor);
   return layer_monitor;
}

Reconstruction_Cost_Monitor *Monitor::construct_RCM(MLP *mlp) {
   Reconstruction_Cost_Monitor *rc_monitor = new Reconstruction_Cost_Monitor(mlp);
   stat_units.push_back(rc_monitor);
   return rc_monitor;
}

Layer_Monitor::Layer_Monitor(MLP *mlp, Layer *layer, Visualizer *viz) : Monitor(viz) {
   threshold = 0;
   layer_monitor = construct_LTDM(mlp, layer);
   layer_monitor->set_coords(0, 2, 0);
   layer_monitor->set_size(8, 4, 0);
   layer_monitor->pack();
   
   rc_monitor = construct_RCM(layer_monitor->mlp);
   rc_monitor->set_coords(0, -5, 0);
   rc_monitor->set_size(8, 1, 0);
   
   class_monitor = NULL;
#ifdef USEGL
   viz->init();
#endif
   load_units_into_visualizer(viz);
}

Layer_Monitor::Layer_Monitor(MLP *mlp, Layer *layer, Layer *class_layer, Visualizer *viz) : Monitor(viz) {
   threshold = 0;
   layer_monitor = construct_LTDM(mlp, layer);
   layer_monitor->set_coords(0, 0, 0);
   layer_monitor->set_size(8, 4, 0);
   layer_monitor->pack();
   
   rc_monitor = construct_RCM(layer_monitor->mlp);
   rc_monitor->set_coords(0, -7, 0);
   rc_monitor->set_size(8, 1, 0);
   
   class_monitor = construct_LTDM(mlp, class_layer);
   class_monitor->set_coords(0, 7, 0);
   class_monitor->set_size(8, .5, 0);
   class_monitor->pack();
   grid_units.push_back(class_monitor);
#ifdef USEGL
   viz->init();
#endif
   load_units_into_visualizer(viz);
}

void Layer_Monitor::save(std::string name) {
   layer_monitor->update();
   if (class_monitor != NULL) class_monitor->update();
   save_features(this, name);
}

