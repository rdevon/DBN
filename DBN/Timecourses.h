//
//  Timecourses.h
//  DBN
//
//  Created by Devon Hjelm on 8/16/12.
//
//

#ifndef __DBN__Timecourses__
#define __DBN__Timecourses__

#include <iostream>

class RBM;
class Connection;
class DataSet;

void get_timecourses(RBM *rbm, Connection* connection, DataSet *data);

#endif /* defined(__DBN__Timecourses__) */
