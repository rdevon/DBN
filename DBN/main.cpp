//
//  main.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/10/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "Params.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_permute.h"
#include "DataSets.h"
#include "Layers.h"
#include "Connections.h"
#include "RBM.h"
#include "Viz.h"
#include "Monitors.h"
#include "DBN.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include "SupportFunctions.h"
#include "Get_Network.h"
#include "IO.h"
#include "Tests.h"
#include "Autoencoder.h"

char* get_command_line(char **begin, char **end, const std::string &option) {
   char ** itr = std::find(begin, end, option);
   if (itr != end && ++itr != end)
   {
      return *itr;
   }
   return 0;
}

bool command_option_exists(char** begin, char** end, const std::string &option)
{
   return std::find(begin, end, option) != end;
}

void print_usage(){
   std::cout << "usage: DBN [-n] [-f config file for 3D h5 loading] [-l filename]" << std::endl;
}

int main (int argc, char * argv[])
{
   
   std::streambuf *psbuf = NULL;
   std::ofstream log_stream;
   std::streambuf *pStreambuf = std::cout.rdbuf();
   
   //--------------RNG INIT STUFF
   srand((unsigned)time(0));
   long seed;
   r = gsl_rng_alloc (gsl_rng_rand48);     // pick random number generator
   seed = time (NULL) * getpid();
   gsl_rng_set (r, seed);                  // set seed
   // TESTING AREA
   
   //---------------
   DBN *dbn;
   
   if (command_option_exists(argv, argv+argc, "-n")) {
      dbn = return_network();
      std::cout << dbn << std::endl;
      std::cout << "Ready to learn" << std::endl;
      std::cout << "If you are using visualization:" << std::endl;
      std::cout << "'V': pauses visualization" << std::endl;
      std::cout << "<SPACE>: pauses learning" << std::endl;
      std::cout << "'+'/'-' increases/decreases learning rate" << std::endl;
      std::cout << "'['/']' increases/decreases output threshold" << std::endl;
      std::cout << "'L' stops learning for layer (skips to next if any)" << std::endl;
      std::cout << "Current visualization is the features displayed on top with the plot of the reconstruction cost in the box (gl text not supported yet)" << std::endl;
      std::cout << "Press <ENTER> to start learning: ";
      std::cin.get();
   }
   else if (command_option_exists(argv, argv+argc, "-l")) {
      std::string filename = get_command_line(argv, argv+argc, "-l");
      MLP mlp = load_MLP(filename);
      dbn = new DBN(mlp);
      dbn->data_layers.clear();
      dbn->view();
      exit(EXIT_SUCCESS);
   }
   else if (command_option_exists(argv, argv+argc, "-f")) {
      if (argc != 3) {
         print_usage();
         exit(EXIT_FAILURE);
      }
      //Initialization, time, logfile stuff, etc.
      time_t t = time(0);   // get time now
      struct tm * the_time = localtime( & t );
      mkdir(out_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      out_path += convert_to_string(*the_time) + "/";
      mkdir((out_path).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      
      std::string filename = get_command_line(argv, argv+argc, "-f");
      dbn = return_aod_network(filename, log_stream, psbuf);
      
   }
   else if (command_option_exists(argv, argv+argc, "-F")) {
      
      std::string filename = get_command_line(argv, argv+argc, "-F");
      MLP mlp = load_MLP(filename); 
      Autoencoder ae(mlp);
      ae.name = (mlp).name + "fine_tuning";
      Gradient_Descent gd(0);
      gd.teachAE(ae);
      MLP ae_mlp;
      save(ae);
      exit(EXIT_SUCCESS);
   }
   else if (command_option_exists(argv, argv+argc, "-stack")) {
      std::string filename = get_command_line(argv, argv+argc, "-stack");
      dbn = load_and_stack(filename, log_stream, psbuf);
   }
   else {
      print_usage();
      exit(EXIT_SUCCESS);
   }
   ContrastiveDivergence cd(1000);
   dbn->learn(cd);
   std::cout.rdbuf(pStreambuf);
   log_stream.close();
   exit(1);
   return 0;
}

