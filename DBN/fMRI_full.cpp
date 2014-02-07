//
//  fMRI_full.cpp
//  DBN
//
//  Created by Devon Hjelm on 11/13/12.
//
//

#include "DataSets.h"

#include <H5Cpp.h>
#include "IO.h"
#include "SupportFunctions.h"
#include <math.h>

DataSet *load_fMRI3D_DS(std::string filename) {
   std::cout << "Opening file " << filename << std::endl;

   typedef DataSet DBN_DS;
   
   using namespace H5;
   using H5::DataSet;
   
   const H5std_string FILE_NAME(filename);
   
   H5File file(FILE_NAME, H5F_ACC_RDONLY);
   
   int dims[4];
   DataSet datadims = file.openDataSet("dims");
   datadims.read(dims, PredType::NATIVE_INT);
   datadims.close();
   
   DataSet data = file.openDataSet("data");
   DataSpace dsp = data.getSpace();
   
   hsize_t dims_out[2];
   dsp.getSimpleExtentDims(dims_out, NULL);
   int volumes = (int)dims_out[0];
   int voxels = (int)dims_out[1];
   
   std::cout << "Dataset images are of size " << dims[0] << "x" << dims[1] << "x" << dims[2] << "x" << dims[3] << std::endl;
   std::cout << "Training set is of size " << volumes << "x" << voxels << std::endl;
   
   DBN_DS *dataset = new DBN_DS(AOD, volumes, dims[0], dims[1], dims[2], voxels);
   dataset->data_path = filename;
   
   std::cout << "Reading dataset" << std::endl;
   float *out_buffer = new float[voxels*volumes];
   data.read(out_buffer, PredType::NATIVE_FLOAT);
   
   for (int volume = 0; volume < volumes; ++volume) {
      for (int voxel = 0; voxel < voxels; ++voxel) {
         float val = out_buffer[volumes*voxel + volume];
         dataset->data(volume,voxel) = val;
      }
   }
   
   data.close();
   dsp.close();
#if 0
   Vector col = transpose(dataset->data)(9);
   col.save("/Users/devon/Research/matlab/tocmat2");
   exit(1);
#endif
   
   std::cout << "Reading mask" << std::endl;
   DataSet mask_data = file.openDataSet("mask");
   dsp = mask_data.getSpace();
   mask_data.read(dataset->mask.m->data, PredType::NATIVE_FLOAT);
   mask_data.close();
   std::cout << "Done reading mask" << std::endl;
   dataset->applymask = true;
   std::cout << "Done loading dataset" << std::endl;
   return dataset;
}

DataSet *load_AOD_stim(run_t run, int dup) {
   float scan_time = 1.5;
   int scans_per_run = 249;
   //float stim_time = .2;
   size_t steps;
   std::vector<float> target,novel,standard;
   switch (run) {
      case RUN2: {
         steps = scans_per_run;
         target = load_dlm(aod_stim_path + "target_run2");
         novel = load_dlm(aod_stim_path + "novel_run2");
         //standard = load_dlm(aod_stim_path + "standard_run2");
      } break;
      case RUN1: {
         steps = scans_per_run;
         target = load_dlm(aod_stim_path + "target_run1");
         novel = load_dlm(aod_stim_path + "novel_run1");
         //standard = load_dlm(aod_stim_path + "standard_run1");
      } break;
      case RUN12 : {
         steps = 2*scans_per_run;
         target = load_dlm(aod_stim_path + "target_run1");
         novel = load_dlm(aod_stim_path + "novel_run1");
         //standard = load_dlm(aod_stim_path + "standard_run1");
         std::vector<float> target2 = load_dlm(aod_stim_path + "target_run2");
         std::vector<float> novel2 = load_dlm(aod_stim_path + "novel_run2");
         //std::vector<float> standard2 = load_dlm(aod_stim_path + "standard_run2");
         for (auto &val:target2) val += scans_per_run*scan_time;
         for (auto &val:novel2) val += scans_per_run*scan_time;
         //for (auto &val:standard2) val += scans_per_run*scan_time;
         target.insert(target.end(), target2.begin(), target2.end());
         novel.insert(novel.end(), novel2.begin(), novel2.end());
         //standard.insert(standard.end(), standard2.begin(), standard2.end());
      }
   }
   DataSet *dataset = new DataSet(AOD_STIM, steps*dup, 0, 0, 0, 2);
   dataset->data_path = convert_to_string(run);
   dataset->data.set_all(0);
   Matrix stim(steps,2);
   for (auto start_time:target) stim(start_time/scan_time,0) += 1;
   for (auto start_time:novel) stim(start_time/scan_time,1) += 1;
   //for (auto start_time:standard) dataset->data(start_time/scan_time,2) += 1;
   
   for (int i = 0; i < dup; ++i) {dataset->data.fill_submatrix(stim, i*(int)steps);}
   
   return dataset;
}