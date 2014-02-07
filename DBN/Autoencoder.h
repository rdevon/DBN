//
//  Autoencoder.h
//  DBN
//
//  Created by Devon Hjelm on 1/5/13.
//
//

#ifndef __DBN__Autoencoder__
#define __DBN__Autoencoder__

#include <iostream>
#include "MLP.h"
#include "Teacher.h"

class Autoencoder : public MLP {
public:
   std::map<Layer*, Layer*> data_map;
   MLP encoder;
   MLP decoder;
   Autoencoder(MLP &mlp);
   void learn(Gradient_Descent&);
};

#endif /* defined(__DBN__Autoencoder__) */
