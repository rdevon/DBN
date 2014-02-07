//
//  IO.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/23/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include "DataSets.h"
#include "SupportFunctions.h"
#include "Viz_Units.h"

#include "Params.h"

void DataSet::zeromean_unitvar_pixel(){
   Vector mean_image = data.mean_image();
   data-=mean_image;
   Vector sd_image = data.sd_image();
   data /= sd_image;
}

void DataSet::removeMask(){
   std::cout << "Removing mask" << std::endl;
   Vector mean_image = data.mean_image();
   mask = mean_image.make_mask();
   data.remove_mask(mask);
   std::cout << "Mask removed" << std::endl;
}

void DataSet::apply_mask(Vector &dest, Vector& src){
   if (applymask) dest = src.add_mask(mask);
   else dest = src;
}

void DataSet::transform_for_viz(Matrix &dest, Vector &src){
   dest.load(src.add_mask(mask));
}

void DataSet::normalize(){
   data.normalize();
}

void DataSet::make_validation(gsl_rng *rand) {
   Matrix temp(data);
   //temp.shuffle_rows(rand);
   train.copy_submatrices(temp, 0);
   validation.copy_submatrices(temp, (int)train.dim1);
   std::cout << "Finished making validation set" << std::endl;
}
