//
//  Viz_Units.h
//  DBN
//
//  Created by Devon Hjelm on 10/23/12.
//
//

#ifndef __DBN__Viz_Units__
#define __DBN__Viz_Units__

#include <iostream>
#include <vector>
#include "Matrix.h"

using std::vector;

class Monitor;
class Visualizer;

class Viz_Unit {
public:
   
   bool                    up_to_date;
   bool                    on;
   
   float                   x_position,
   y_position,
   z_position;
   float                   x_size,
   y_size,
   z_size;
   
   float                   color[4];
   float                   _max, _min;
   
   void set_coords(float x, float y, float z);
   void set_size(float x, float y, float z) {x_size = x, y_size = y, z_size = z;}
   Viz_Unit(){}
   
   virtual void update() = 0;
   virtual void load_into_visualizer(Visualizer *viz) = 0;
   virtual void scale(float scale_factor);
   virtual void normalize(float min = 0, float max = 0) = 0;
};

class Tex_Unit : public Viz_Unit {
public:
   
   float                   threshold;
   Matrix                  viz_matrix;
   Vector                  stat_vector;
   Tex_Unit(size_t dim1, size_t dim2) : viz_matrix(dim1, dim2) {}
   
   void normalize(float min = 0, float max = 0);
   virtual void update() = 0;
   void load_into_visualizer(Visualizer *viz);
   void clear();
   void apply_threshold(Monitor* = NULL);
};

class Stacked_Tex_Unit : public Tex_Unit {
public:
   int                                    level, stack_top, stack_bot;
   
   Stacked_Tex_Unit(size_t dim1, size_t dim2) : Tex_Unit(dim1, dim2){}
   virtual void update() = 0;
   void clear();
   void step_forward();
   void step_back();
};

class Grid_Viz_Unit : public Viz_Unit {
public:
   float threshold;
   std::vector<Viz_Unit*>           units;
   
   Grid_Viz_Unit(){}
   void pack();
   void scale_matrix_and_threshold(Monitor* = NULL);
   virtual void update();
   void load_into_visualizer(Visualizer *viz);
   void clear();
   void scale(float scale_factor);
   void normalize(float min = 0, float max = 0);
};

class Multi_Unit : public Viz_Unit {
public:
   std::vector<Viz_Unit*>           units;
   
   Multi_Unit(){}
   void pack();
   void update();
   void normalize(float min = 0, float max = 0){}
   void load_into_visualizer(Visualizer *viz);
   void clear();
   void scale(float scale_factor);
};

class Plot_Unit : public Viz_Unit {
public:
   Vector                           line_set;
   Plot_Unit(size_t dim) : line_set(dim) {}
   virtual void update() = 0;
   void load_into_visualizer(Visualizer *viz);
   void clear();
   void normalize(float min = 0, float max = 0);
};

#endif /* defined(__DBN__Viz_Units__) */
