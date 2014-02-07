//
//  SPM.cpp
//  DBN
//
//  Created by Devon Hjelm on 11/13/12.
//
//

#include "DataSets.h"
#if 0
void DataSet::loadSPM(){
   dims[3] = 220;
   dims[0] = 2;
   dims[1] = 1;
   dims[2] = 1;
   train = gsl_matrix_float_alloc(dims[3], dims[1]*dims[0]);
   std::cout << "Loading stimulus" << std::endl;
   
   std::string filename, pathname;
   std::ifstream file;
   
   int sample = 0;
   
   pathname = SPMpath + "SPM.dat";
   
   file.open(pathname.c_str());
   
   std::string line;
   while (getline(file, line)){
      float value;
      int index = 1;
      std::istringstream iss(line);
      while (iss >> value) {
         if (index == 1 || index == 2)
            for (int i = 0; i < dims[0]/2; ++i){
               gsl_matrix_float_set(train, sample, index-1+(2*i), value);
            }
         ++index;
      }
      ++sample;
   }
   file.close();
}
#endif