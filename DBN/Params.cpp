//
//  Types.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/29/12.
//
//

#include "Params.h"
#include "SupportFunctions.h"

//std::string path = "/home/pliz/soft/src/dev/tools/DBN-2.0/";
#if __APPLE__
std::string path = "/Users/devon/Research/DBN_current/";
std::string exec_path = "/Users/devon/Research/DBN_current";
#else
std::string path = std::string(SOURCE_DIR) + std::string("/../");
std::string exec_path = std::string(BIN_DIR) + std::string("/");
#endif

#if __APPLE__
std::string out_path = "/Users/devon/Research/MLPs/";
#else
std::string out_path = std::string(OUT_DIR) + std::string("/MLPs/");
#endif

gsl_rng * r;
std::string MNISTpath = path + "../MNISTdata/";
std::string plotpath = path + "plots/";
std::string fMRIpath = path + "../fMRIdata";
std::string SPMpath = path + "../SPMdata/";
std::string vertexPath = path + "Shaders/Vertex.c";
std::string fragmentPath = path + "Shaders/Fragment.c";
std::string aod_path = path + "../healthy_run1_one_sn2.h5";
std::string aod_stim_path = path + "../AOD_stims/";
std::string fMRI_3D_path = aod_path;

std::ostream &operator<< (std::ostream &os, const struct tm the_time){
   os << convert_to_string(the_time.tm_mon+1);
   os << "_";
   os << convert_to_string(the_time.tm_mday);
   os << "_";
   os << convert_to_string(the_time.tm_year - 100);
   return os;
}

std::ostream &operator<< (std::ostream &os, const Layer_t lt) {
   switch (lt) {
      case SIGMOID: os << "Sigmoid"; break;
      case GAUSSIAN: os << "Gaussian"; break;
      case RELU: os << "ReLU"; break;
      case SOFTMAX: os << "Sotfmax"; break;
   }
   return os;
}

std::ostream &operator<< (std::ostream &os, const Dataset_t dst) {
   switch (dst) {
      case MNIST: os << "MNIST"; break;
      case MNIST_L: os << "MNIST_lables"; break;
      case SSL_VIS: os << "SS_visuomotor"; break;
      case VIS_STIM: os << "Visuomotor_stim"; break;
      case VOL_VIS: os << "3D_visuomotor"; break;
      case AOD: os << "AOD"; break;
      case AOD_STIM: os << "AOD_stim"; break;
      case NA: os << "NA"; break;
   }
   return os;
}