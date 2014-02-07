//
//  Monitors.h
//  DBN
//
//  Created by Devon Hjelm on 10/23/12.
//
//

#ifndef __DBN__Monitors__
#define __DBN__Monitors__

#include <iostream>
#include <vector>
#include <map>

class Stacked_Tex_Unit;
class Visualizer;
class Teacher;
class Plot_Unit;
class Tex_Unit;
class Viz_Unit;
class fMRI_Layer_Monitor;
class Reconstruction_Cost_Monitor;
class MLP;
class DBN;
class MNIST_Layer_Monitor;
class Simple_3D_Monitor;
class DataSet;
class Stacked_Tex_Unit;
class Grid_Viz_Unit;
class Layer_to_Data_Monitor;
class Layer;

using std::vector;

class Monitor {
public:
   Teacher                          *teacher;
   Visualizer                       *viz;
   bool                             recvd_press;
   
   std::vector<Viz_Unit*>           single_units;
   std::vector<Stacked_Tex_Unit*>   stacked_units;
   std::vector<Grid_Viz_Unit*>      grid_units;
   std::vector<Viz_Unit*>           stat_units;
   
   float                            threshold;
   
   Monitor();
   Monitor(Visualizer *viz);
   void load_units_into_visualizer(Visualizer *);
   void update();
   void peek();
   void update_stats();
   void send_stop_signal();
   void toggle_monitor_top();
   void move_up_stack();
   void move_down_stack();
   void view();
   virtual void save(std::string name){}
   
   Layer_to_Data_Monitor *construct_LTDM(MLP *mlp, Layer *layer);
   Reconstruction_Cost_Monitor *construct_RCM(MLP *mlp);

};

class Layer_Monitor : public Monitor {
public:
   
   Layer_to_Data_Monitor         *layer_monitor;
   Layer_to_Data_Monitor         *timecourse_monitor;
   
   Reconstruction_Cost_Monitor   *rc_monitor;
   Layer_to_Data_Monitor         *class_monitor;
   
   Layer_Monitor(MLP* mlp, Layer *feature_layer, Visualizer *viz);
   Layer_Monitor(MLP* mlp, Layer *feature_layer, Layer *class_layer, Visualizer *viz);
   
   void save(std::string name);
};


#endif /* defined(__DBN__Monitors__) */
