//
//  Connections.h
//  DBN
//
//  Created by Devon Hjelm on 8/15/12.
//
//

#ifndef __DBN__Connections__
#define __DBN__Connections__

#include <iostream>
#include "MLP.h"
#include "Teacher.h"
#include "Viz.h"
#include "Matrix.h"

class Layer;

/////////////////////////////////////
// Connection class
/////////////////////////////////////

// Binary connections between layers.  This is where the propagation happens as well as being a container for
// layers.  Layers are kept inside binary connections.  This should be make it easy to make generic graphs
// of layers.
class Connection : public LearningUnit {
public:
   float freeEnergy;
   float transmission_scale;
   Matrix weights;
   Layer *from, *to;
   
   Connection(Layer *from, Layer *to);
   Connection(const Connection& other, Layer *from, Layer *to, bool trans) : LearningUnit(trans? other.weights.dim2 : other.weights.dim1, trans? other.weights.dim1 : other.weights.dim2), weights(trans ? transpose(other.weights) : other.weights), transmission_scale(trans ? 1/other.transmission_scale : other.transmission_scale), from(from), to(to) {}
   void reset(){weights.set_gaussian(0.01);}
   
   Transmit_t transmit_signal(Direction_t direction, Purpose_t purpose, Sample_flag_t s_flag);
   void transmit_data(const Matrix &from, Matrix &to);
   
   void catch_stats(Stat_flag_t);
   void update(Teacher&, bool);
   void set_defaults();
   
   void getFreeEnergy();
};

#endif /* defined(__DBN__Connections__) */