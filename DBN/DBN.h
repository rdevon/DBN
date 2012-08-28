//
//  DBN.h
//  DBN
//
//  Created by Devon Hjelm on 8/27/12.
//
//

#ifndef __DBN__DBN__
#define __DBN__DBN__

#include <iostream>

class DBNLayer{
public:
   DBNLayer *up;
   DBNLayer(){up = NULL;}
};

class DBN{
public:
   DBNLayer *bottom;
   void learn();
};

#endif /* defined(__DBN__DBN__) */
