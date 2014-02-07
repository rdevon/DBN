//
//  Types.h
//  DBN
//
//  Created by Devon Hjelm on 7/11/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_Types_h
#define DBN_Types_h

#include <algorithm>
#include <stdlib.h>
#include "gsl/gsl_rng.h"
#include <gsl/gsl_randist.h>
#include <vector>
#include <string>
#include <map>
#include <ctime>

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include "dirent.h"
#include <sstream>
#include <gsl/gsl_statistics.h>
#define PI 3.1415


typedef int lid_t; //layer id type
typedef int cid_t; //connection id type
typedef int did_t; //dataset id type
typedef int iid_t; //input edge id type

extern std::string path;
extern std::string exec_path;
extern std::string MNISTpath;
extern std::string plotpath;
extern std::string fMRIpath;
extern std::string SPMpath;
extern std::string vertexPath;
extern std::string fragmentPath;
extern std::string fMRI_3D_path;
extern std::string out_path;
extern std::string aod_path;
extern std::string aod_stim_path;
extern gsl_rng * r;

typedef enum{UP, DOWN} Direction_t;
inline Direction_t operator!(Direction_t dir)
{
   switch (dir) {
      case UP:       return DOWN;
      case DOWN:     return UP;
   }
}

typedef enum {SAMPLE, NOSAMPLE} Sample_flag_t;

typedef enum {NEG, POS} Stat_flag_t;

typedef enum {WHITE = -100, GREY, BLACK, BLUE, RED, GREEN, YELLOW} Color_t;

typedef enum {SIGMOID = 1, GAUSSIAN, RELU, SOFTMAX} Layer_t;
std::ostream &operator<< (std::ostream &os, const Layer_t lt);

typedef enum {MNIST = 1, MNIST_L, SSL_VIS, VOL_VIS, AOD, AOD_STIM, VIS_STIM, NA} Dataset_t;
std::ostream &operator<< (std::ostream &os, const Dataset_t dst);

typedef enum {AUTO = -1} Param_t;

typedef enum {L1NORM, L2NORM, MAXWEIGHT, NONE} Decay_t;

typedef enum {DONE, OK, SLOW, BROKEN} rc_prog_t;

typedef enum {LEARNING, GENERATING, TESTING, RECOGNITION, VISUALISATION} Purpose_t;

typedef enum {TRANSMIT_SUCCESS, TRANSMIT_FAILURE} Transmit_t;

std::ostream &operator<< (std::ostream &os, const struct tm the_time);



#endif
