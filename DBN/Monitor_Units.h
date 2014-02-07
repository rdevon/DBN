//
//  File.h
//  DBN
//
//  Created by Devon Hjelm on 10/23/12.
//
//

#ifndef __DBN__File__
#define __DBN__File__

#include <iostream>
#include <vector>
#include "Viz_Units.h"
#include "Params.h"

using std::vector;

class DataSet;
class MNIST_Feature_Monitor;
class MLP;
class Layer;
class Feature_to_Data_Monitor;
class fMRI_Feature_Monitor;
class Reconstruction_Cost_Monitor;
class Static_Tex;
class Input_Edge;

class Static_Tex : public Tex_Unit {
public:
   Static_Tex(Matrix tex);
   void update(){}
};

class Layer_to_Data_Monitor : public Grid_Viz_Unit {
public:
   MLP                           *mlp;
   Layer                         *feature_layer;
   
   Layer_to_Data_Monitor(MLP *mlp, Layer *feature_layer);
   void update();
};

class Feature_to_Data_M_Unit : public Stacked_Tex_Unit {
public:
   int                     feature;
   
   MLP                     *mlp;
   Layer                   *feature_layer;
   
   Feature_to_Data_M_Unit(int feat, MLP *path, Layer *feature_layer);
   void update();
};

class Timecourse_M_Unit : public Plot_Unit {
public:
   int feature;
   
   MLP                     *mlp;
   Layer                   *feature_layer;
   
   Timecourse_M_Unit(int feat, MLP *path, Layer *feature_layer);
   void update();
};

class Reconstruction_Cost_Monitor : public Plot_Unit {
public:
   MLP                     *mlp;
   int epoch;
   rc_prog_t status;
   Reconstruction_Cost_Monitor(MLP *mlp, int epochs = 100);
   void update();
   void check();
};

#endif /* defined(__DBN__File__) */
