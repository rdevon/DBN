//
//  fMRI_Slice.cpp
//  DBN
//
//  Created by Devon Hjelm on 11/10/12.
//
//

#include "DataSets.h"

DataSet *load_SS_fMRI_DS(std::string pathname){
   
   std::cout << "Loading fMRI" << std::endl;
   
   std::string filename, datapath;
   
   struct dirent *filep;
   struct stat filestat;
   std::ifstream file;
   DIR *dir;
   
   std::string path;
   Matrix data(220, 63*53);
   for (int s = 1; s <=1; ++s) {
      std::stringstream n;
      n << s;
      path = pathname + n.str() + "/";
      dir = opendir(path.c_str());
      Matrix subject(220, 63*53);
      int sample = 0;
      while ((filep = readdir(dir))){
         filename = filep->d_name;
         datapath = path + filep->d_name;
         
         // If the file is a directory (or is in some way invalid) we'll skip it
         if (stat( datapath.c_str(), &filestat )) continue;
         if (S_ISDIR( filestat.st_mode ))         continue;
         if (filename == ".DS_Store")             continue;
         
         //cout << "Loading " << filename << endl;
         
         file.open(datapath.c_str());
         
         std::string line;
         int index = 0;
         while (getline(file, line)){
            float value;
            std::istringstream iss(line);
            while (iss >> value) {
               subject(sample, index) = value;
               ++index;
            }
         }
         file.close();
         ++sample;
      }
      closedir(dir);
      //subject.normalize();
      data.fill_submatrix(subject, (s-1)*200);
   }
   Vector meanImage = data.mean_image();
   Vector mask = meanImage.make_mask();
   data.remove_mask(mask);
   
   DataSet *dataset = new DataSet(SSL_VIS, 220, 53, 63, 1, data.dim2);
   dataset->data_path = pathname;
   dataset->data = data;
   dataset->mask = mask;
   
   Vector mi = dataset->data.sd_image();
   dataset->zeromean_unitvar_pixel();
   
   dataset->applymask = true;
   return dataset;
}


DataSet *load_FMRI_S(){
   DataSet *dataset = new DataSet(VIS_STIM, 220, 2, 1, 1);
   std::cout << "Loading stimulus" << std::endl;
   dataset->data.set_all(0);
   for (int i = 0; i < 220; ++i) {
      if (i%55 < 15)
         dataset->data(i,0) = 1;
      else if ((i - 20)%55 < 15 && i >= 20)
         dataset->data(i,1) = 1;
   }
   return dataset;
}
#if 0
void DataSet::loadstim2(){
   name = "fMRI_stim";
   type = FMRI_STIM;
   dims[3] = 220;
   dims[0] = 3;
   dims[1] = 1;
   dims[2] = 1;
   train = gsl_matrix_float_calloc(dims[3], dims[0]*dims[1]);
   std::cout << "Loading stimulus" << std::endl;
   
   for (int i = 0; i < 220; ++i) {
      if (i%55 < 15)
         gsl_matrix_float_set(train, i, 0, 1);
      else if ((i - 20)%55 < 15 && i >= 20)
         gsl_matrix_float_set(train, i, 1, 1);
      else
         gsl_matrix_float_set(train, i, 2, 1);
   }
}
#endif