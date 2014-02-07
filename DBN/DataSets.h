//
//  IO.h
//  DBN
//
//  Created by Devon Hjelm on 7/23/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_IO_h
#define DBN_IO_h
#include <stdint.h>
#include <arpa/inet.h>
#include "Params.h"
#include <H5Cpp.h>
#include "Matrix.h"

typedef enum {RUN1, RUN2, RUN12} run_t;
inline std::ostream &operator<< (std::ostream &os, const run_t lt) {
   switch (lt) {
      case RUN1: {
         os << "run1";
         } break;
      case RUN2: {
         os << "run2";
      } break;
      case RUN12: {
         os << "run12"; break;
      } break;
         }
         return os;
}

class Layer;
class Viz_Unit;

class DataSet{
public:
   Dataset_t type;
   std::string data_path;
   int masksize, index;
   size_t dims[4];
   float validation_perc;
   
   bool applymask;
   
   Matrix data, validation, train;
   Vector mask, image;
   Viz_Unit         *viz_unit;
   
   DataSet(size_t data_number, size_t data_size) :
   validation_perc(.1), type(NA), data(data_number, data_size),
   validation(validation_perc*data_number, data_size), train((size_t)(data_number-validation.dim1), data_size),
   image(0), mask(0), dims{0,0,0,data_number}, masksize(0), applymask(false), data_path("") {}
   
   DataSet(Dataset_t dst, size_t t = 0, size_t x = 0, size_t y = 0, size_t z = 0, size_t data_size = 0) :
   validation_perc(.1), type(dst), data(t, data_size ? data_size : x*y*z), validation((validation_perc*t), data_size ? data_size : x*y*z),
   train((size_t)(t - validation.dim1), data_size ? data_size : x*y*z), image(x*y*z),
   mask(x*y*z),dims{x,y,z,t}, masksize(0), applymask(false), data_path("") {}
   
   void splitValidate(float percentage = .1);
   void zeromean_unitvar_pixel();
   void getMask();
   void removeMask();
   void transform_for_viz(Matrix &dest, Vector &src);
   void apply_mask(Vector &dest, Vector &src);
   void normalize();
   void make_validation(gsl_rng *rand = r);
};

DataSet *load_MNIST_DS(std::string filename = MNISTpath);
DataSet *load_MNIST_L(std::string filename = MNISTpath);
DataSet *load_SS_fMRI_DS(std::string path = fMRIpath);
DataSet *load_fMRI3D_DS(std::string filename = fMRI_3D_path + "visuomotor_data.h5");
DataSet *load_FMRI_S();
DataSet *load_AOD_stim(run_t run, int dup = 1);

#endif
