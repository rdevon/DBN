//
//  Viz_Units.cpp
//  DBN
//
//  Created by Devon Hjelm on 10/23/12.
//
//

#include "Viz_Units.h"
#include "Monitors.h"
#include "Params.h"
#include "SupportFunctions.h"
#include "Viz.h"
#include <math.h>
 
// FOR UNIT MONITORS -----------------------------------------------------------------

void Viz_Unit::set_coords(float x, float y, float z) {x_position = x, y_position = y, z_position = z;}

void Viz_Unit::scale(float scale_factor) {
   x_size *= scale_factor;
   y_size *= scale_factor;
   z_size *= scale_factor;
}

void Tex_Unit::clear(){viz_matrix.set_all(0);}

void Plot_Unit::clear(){line_set.set_all(0);}

void Tex_Unit::load_into_visualizer(Visualizer *viz) {
#ifdef USEGL
   viz->add_tex(this);
#endif
}

void Plot_Unit::load_into_visualizer(Visualizer *viz) {
#ifdef USEGL
   viz->add_plot(this);
#endif
}

void Stacked_Tex_Unit::step_forward() { if (level < stack_top) ++level; }

void Stacked_Tex_Unit::step_back() { if (level > stack_bot) --level; }

void Tex_Unit::normalize(float min, float max) {
   min = min ? min : _min;
   max = max ? max : _max;
   viz_matrix.apply([&] (float &x) {
      if (x > 0 && min > 0) {
         x-=min;
         x/=(max-min);
      }
      else if (x > 0 && min <= 0) x/=max;
      else if (x != WHITE) x/=(-min);}); }

void Grid_Viz_Unit::normalize(float min, float max) {
   for (auto unit:units) {
      if (unit->_min < min) min = unit->_min;
      if (unit->_max > max) max = unit->_max;
   }
   for (auto unit:units) unit->normalize(min,max);
}

void Grid_Viz_Unit::pack() {
   float unit_area = (units[0]->x_size) * (units[0]->y_size);
   float total_area = (x_size*y_size);
   
   float scale_factor = sqrt(total_area/(unit_area * units.size()));
   
   int number = 0;
   
   for (auto unit:units) {
      unit->scale(scale_factor);
      
      int x_pos = number%(int)(x_size/unit->x_size);
      int y_pos = number/(int)(x_size/unit->x_size);
   
      float x_ref = x_position-x_size + unit->x_size;
      float y_ref = y_position+y_size;
   
      unit->set_coords(x_ref + 2*x_pos*unit->x_size, y_ref-2*y_pos*unit->y_size, 0);
      
      ++number;
   }
}

void Grid_Viz_Unit::update() {
   for (auto unit:units) unit->update();
   normalize();
}

void Grid_Viz_Unit::load_into_visualizer(Visualizer *viz) {for (auto unit:units) unit->load_into_visualizer(viz);}

void Grid_Viz_Unit::scale(float scale_factor) {for (auto unit:units) unit->scale(scale_factor);}

void Multi_Unit::update() {
   for (auto unit:units) unit->update();
   pack();
}

void Multi_Unit::load_into_visualizer(Visualizer *viz) {for (auto unit:units) unit->load_into_visualizer(viz);}

void Multi_Unit::pack() {
   float x_extent = 0;
   for (auto unit:units) if (unit->x_size > x_extent) x_extent = unit->x_size;
   for (auto unit:units) unit->scale(x_extent/unit->x_size);
   scale(x_size/x_extent);
   float y_extent = 0;
   for (auto unit:units) y_extent += unit->y_size;
   if (y_extent > y_size)
      scale(y_size/y_extent);
   
   float x_pos = x_position;
   float y_pos = y_position-y_size/2;
   for (auto unit:units) {
      y_pos += unit->y_size/2;
      unit->set_coords(x_pos, y_pos, 0);
      y_pos += unit->y_size/2;
   }
}

void Multi_Unit::scale(float scale_factor) {for (auto unit:units) unit->scale(scale_factor);}

void Plot_Unit::normalize(float min, float max) {
   line_set.min_max(_min, _max);
   line_set -= _min;
   line_set /= (_max - _min);
}