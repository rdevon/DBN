//
//  DBN.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/27/12.
//
//

#include "DBN.h"
#include "Connections.h"
#include "RBM.h"
#include "Layers.h"
#include "DataSets.h"
#include "Monitors.h"
#include "IO.h"
#include "Autoencoder.h"

DBN::DBN(MLP reference_MLP) : MLP(), reference_MLP(reference_MLP) {viz_layer = reference_MLP.viz_layer; data_layers = reference_MLP.data_layers;}

void DBN::learn(ContrastiveDivergence &teacher){
   std::cout << "Begining DBN Learning" << std::endl;
   int l = (int)levels.size();
#ifdef USEGL
   Visualizer *viz;
   if (monitor_dbn) viz = new Visualizer(1000,1000);
#else
   Visualizer *viz = new Visualizer();
#endif
   std::cout << "DBN: " << std::endl << *this;
   std::cout << "Learning MLP: " << std::endl << reference_MLP;
   Level level = reference_MLP.levels[l];
   add(level);
   while (1){
      RBM rbm(level);
      clock_t lStart = clock();
      
      if (viz_layer) {
         Monitor *monitor = new Layer_Monitor(this, level.top_layers[0], viz);
         teacher.monitor = monitor;
         monitor->teacher = &teacher;
      }
      else std::cout << "Warning, no visualization layer" << std::endl;
      
      rbm.learn(teacher);
      std::cout << "Done training " << level << " layer." << std::endl << "Layer took " << (double)(clock()-lStart)/CLOCKS_PER_SEC << " seconds to train" << std::endl;
      
      // Next level
      ++l;
      if (l == reference_MLP.levels.size()) break;
      Level new_level = reference_MLP.levels[l];
      add(new_level);
      level.transport_data(new_level);
      level = new_level;
   }
   save(*this);
#if 1
   Autoencoder ae(*this);
   ae.name = this->name + ".ft";
   Gradient_Descent gd(0);
   gd.teachAE(ae);
   save(ae);
#endif
#ifdef USEGL
   if (monitor_dbn) viz->close_window();
#endif
}

void DBN::view() {
   Monitor *monitor;
#ifdef USEGL
   Visualizer *viz = new Visualizer(1000,1000);
#else
   Visualizer *viz = new Visualizer();
#endif
   monitor = new Layer_Monitor(this, levels.back().top_layers[0], viz);
   monitor->view();
}

void DBN::stack(Level &new_level) {
   levels.clear();
   data_layers.clear();
   for (auto level:reference_MLP.levels) add(level);
   reference_MLP.add(new_level);
}

