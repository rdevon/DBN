//
//  IO.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/23/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "IO.h"

void DataSet::loadMNIST(){
   
   std::cout << "Loading MNIST" << std::endl;
   
   name = "MNIST";
   
   meanImage_ = NULL;
   mask_ = NULL;
   
   std::string trainfile = MNISTpath + "train-images.idx3-ubyte";
   std::string testfile = MNISTpath + "t10k-images.idx3-ubyte";
   FILE *file_handle;
   file_handle = fopen(trainfile.c_str(), "rb");
   
   unsigned char pixel; 
   uint32_t magicnumber, imageNum, rowNum, colNum;
   
   fread(&magicnumber, sizeof(magicnumber), 1, file_handle);
   fread(&imageNum, sizeof(imageNum), 1, file_handle);
   fread(&rowNum, sizeof(rowNum), 1, file_handle);
   fread(&colNum, sizeof(colNum), 1, file_handle);
   
   
   magicnumber = ntohl(magicnumber);
   imageNum = ntohl(imageNum);
   rowNum = ntohl(rowNum);
   colNum = ntohl(colNum);
   
   train = gsl_matrix_float_alloc(imageNum, colNum*rowNum);
   
   height=rowNum, width = colNum, number = imageNum;
   
   for (int i = 0; i < imageNum; ++i)
      for (int j = 0; j < rowNum*colNum; ++j){
         fread(&pixel, sizeof(pixel), 1, file_handle);
         int val = (int)pixel;
         gsl_matrix_float_set(train, i, j, (float)val);
      }
   
   fclose(file_handle);
   file_handle = fopen(testfile.c_str(), "rb");
   
   fread(&magicnumber, sizeof(magicnumber), 1, file_handle);
   fread(&imageNum, sizeof(imageNum), 1, file_handle);
   fread(&rowNum, sizeof(rowNum), 1, file_handle);
   fread(&colNum, sizeof(colNum), 1, file_handle);
   
   magicnumber = ntohl(magicnumber);
   imageNum = ntohl(imageNum);
   rowNum = ntohl(rowNum);
   colNum = ntohl(colNum);
   
   test = gsl_matrix_float_alloc(imageNum, colNum*rowNum);
   
   for (int i = 0; i < imageNum; ++i)
      for (int j = 0; j < rowNum*colNum; ++j){
         fread(&pixel, sizeof(pixel), 1, file_handle);
         int val = (int)pixel;
         gsl_matrix_float_set(test, i, j, (float)val);
      }
   
   fclose(file_handle);
   
   applymask = false;
   
   std::cout << "Done loading" << std::endl;
   
   denorm = false;
}

void DataSet::loadfMRI(){
   
   number = 220;
   width = 53;
   height = 63;
   
   train = gsl_matrix_float_alloc(number, height*width);
   
   std::cout << "Loading fMRI" << std::endl;
   
   name = "fMRI";
   
   std::string filename, pathname;
   
   struct dirent *filep;
   struct stat filestat;
   std::ifstream file;
   
   DIR *dir = opendir(fMRIpath.c_str());
   
   int sample = 0;
   
   while ((filep = readdir(dir))){
      filename = filep->d_name;
      pathname = fMRIpath + filep->d_name;
      
      // If the file is a directory (or is in some way invalid) we'll skip it
      if (stat( pathname.c_str(), &filestat )) continue;
      if (S_ISDIR( filestat.st_mode ))         continue;
      if (filename == ".DS_Store")             continue;
      
      //cout << "Loading " << filename << endl;
      
      file.open(pathname.c_str());
      
      std::string line;
      int  index  = 0;
      while (getline(file, line)){
         float value;
         std::istringstream iss(line);
         while (iss >> value) {
            gsl_matrix_float_set(train, sample, index, value);
            ++index;
         }
      }
      file.close();
      ++sample;
   }
   closedir(dir);
   
   removeMeanImage();
   removeMask();
   
   applymask = true;
   
   extra = gsl_matrix_float_alloc(train->size1, train->size2);
   gsl_matrix_float_memcpy(extra, train); //This is for time courses since I don't preserve data order in training.
}


void DataSet::removeMeanImage(){
   meanImage_ = gsl_vector_float_calloc(train->size2);
   gsl_vector_float *sample = gsl_vector_float_alloc(train->size2);
   for (int i = 0; i < train->size1; ++i){
      gsl_matrix_float_get_row(sample, train, i);
      gsl_vector_float_add(meanImage_, sample);
   }
   gsl_vector_float_scale(meanImage_, (float)1/(float)train->size1);
   
   for (int i = 0; i < train->size1; ++i){
      gsl_matrix_float_get_row(sample, train, i);
      gsl_vector_float_sub(sample, meanImage_);
      gsl_matrix_float_set_row(train, i, sample);
   }
   
   gsl_vector_float_free(sample);
}

void DataSet::removeMask(){
   int count = 0;
   float mean = gsl_stats_float_mean(meanImage_->data, meanImage_->stride, meanImage_->size);
   for (int i = 0; i < meanImage_->size; ++i) count += (gsl_vector_float_get(meanImage_, i) > mean);
   
   gsl_matrix_float *newtrain = gsl_matrix_float_alloc(number, count);
   
   mask_ = gsl_vector_float_alloc(height*width);
   masksize_ = height*width-count;
   
   for (int j = 0; j < height*width; ++j){
      float val = gsl_vector_float_get(meanImage_, j);
      if (val > mean) gsl_vector_float_set(mask_, j, 1);
      else gsl_vector_float_set(mask_, j, 0);
   }
   
   for (int i = 0; i < number; ++i){
      int jprime = 0;
      for (int j = 0; j < height*width; ++j){
         float maskval = gsl_vector_float_get(mask_, j);
         if (maskval == 1) {
            float val = gsl_matrix_float_get(train, i, j);
            gsl_matrix_float_set(newtrain, i, jprime, val);
            ++jprime;
         }
      }
   }
   
   gsl_matrix_float_free(train);
   train = newtrain;
}

gsl_vector_float *DataSet::applyMask(gsl_vector_float *v){
   gsl_vector_float *newv = gsl_vector_float_calloc(v->size + masksize_);
   for (int i = 0, iprime = 0; i < newv->size; ++i) {
      float maskval = gsl_vector_float_get(mask_, i);
      if (maskval == 1) {
         float val = gsl_vector_float_get(v, iprime);
         gsl_vector_float_set(newv, i, val);
         ++iprime;
      }
   }
   return newv;
}