//
//  RBM.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/19/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include "SupportFunctions.h"
#include "RBM.h"
#include "Connections.h"
#include "Layers.h"
#include "DataSets.h"
#include "Monitors.h"
#include "Monitor_Units.h"
#include "SupportMath.h"

RBM::RBM(Level level) : Level(level) {}

void RBM::reset() {
   for (auto layer:top_layers) layer->reset();
   for (auto connection:connections) connection->reset();
}

void RBM::learn(ContrastiveDivergence& teacher){
   teacher.teachRBM(*this);
   for (auto layer:top_layers) layer->rec_biases = layer->biases;
}

void RBM::gibbs_HV(){
   transmit(UP, LEARNING, NOSAMPLE);
   transmit(DOWN, LEARNING, NOSAMPLE);
}

void RBM::gibbs_VH(){
   for (auto layer:top_layers) layer->sample();
   transmit(DOWN, LEARNING, NOSAMPLE);
   transmit(UP, LEARNING, NOSAMPLE);
}

void RBM::update(ContrastiveDivergence &cd, bool apply_gain){
   for (auto connection:connections) connection->update(cd, apply_gain);
   for (auto layer:top_layers) {
      layer->update(cd, false);
   }
   for (auto layer:bot_layers) {
      layer->update(cd, false);
   }
}

void RBM::catch_stats(Stat_flag_t stat_flag){
   for (auto connection:connections) connection->catch_stats(stat_flag);
   for (auto layer:top_layers) { layer->catch_stats(stat_flag); }
   for (auto layer:bot_layers) { layer->catch_stats(stat_flag); }
}

void RBM::init() {
   for (auto layer:bot_layers) layer->reset();
   int train_size = (int)bot_layers[0]->data->train.dim1;
   for (auto connection:connections) {
      connection->to->set_defaults();
      connection->from->set_defaults();
      Layer *from = connection->from;
      Layer *to = connection->to;
      to->set_with_fanin(*from);
      from->set_with_fanin(*to);
      connection->learning_rate = to->learning_rate;
      
      if (connection->from->data->type == AOD_STIM) {connection->learning_rate = 0.0000001;}
   }
   for (auto layer:bot_layers) layer->learning_rate/=train_size;
   for (auto layer:top_layers) layer->learning_rate/=train_size;
}

float RBM::get_reconstruction_cost(){
   std::cout << "Generating RBM reconstruction cost" << std::endl;
   init_data();
   pull_data(TESTING);
   transmit(UP, TESTING, NOSAMPLE);
   
   std::vector<Matrix> data_mats;
   for (auto connection:connections) data_mats.push_back(connection->from->m_testing);
   
   transmit(DOWN, TESTING, NOSAMPLE);
   
   float reconstruction_cost = 0;
   
   auto d_iter = data_mats.begin();
   for (auto connection:connections) {
      float rc = connection->from->reconstructionCost(*d_iter, connection->from->m_testing);
      std::cout << "Reconstruction cost for [" << *connection << "]: " << rc << std::endl;
      reconstruction_cost += rc;
      ++d_iter;
   }
   
   std::cout << "RBM Reconstruction cost: " << reconstruction_cost << std::endl;
   return reconstruction_cost;
}

//------ RBM Learning Methods

void ContrastiveDivergence::getStats(RBM &rbm) {
   rbm.catch_stats(POS);
   for(int g = 0; g < k; ++g) rbm.gibbs_VH();
   rbm.catch_stats(NEG);
}

void ContrastiveDivergence::teachRBM(RBM &rbm){
   rbm.init();
   
   momentum = .5, k = 1, learning_multiplier = 5;
   std::cout << "Teaching RBM with parameters " << " K = " << k  << ", Momentum = " << momentum << std::endl;
   learning = true;
   int epoch = 0;
   while (epoch <= epochs && learning){
      clock_t eStart = clock();
      //std::cout << "    Epoch " << epoch << std::endl;
      rbm.init_data();
      
      bool data_reset;
      learning_count = 0;
      do {
         std::cout.flush();
         data_reset = !((bool)rbm.pull_data(LEARNING));
         rbm.transmit(UP, LEARNING, NOSAMPLE);
         getStats(rbm);
         rbm.update(*this, data_reset);
#if 0
         if (learning_count%10 == 0) {
            std::cout << "..";
#ifdef USEGL
            if (monitor != NULL) monitor->update();
#endif
         }
#endif
         learning_count += 1;
      } while ( !data_reset );
      ++epoch;
      //std::cout << std::endl;
      
      if (epoch%1 == 0) {
         if (monitor != NULL) {
            //rbm.get_reconstruction_cost();
            std::cout << epoch << ": ";
            monitor->update_stats();
            check_early_stop(&rbm, epoch);
         }
         else rbm.get_reconstruction_cost();
      }
      
      if (learning_multiplier > 1) learning_multiplier *= .8;
      else learning_multiplier = 1;
      //std::cout << "Epoch took " << (double)(clock()-eStart)/CLOCKS_PER_SEC << " seconds" << std::endl;
   }
}

void Teacher::check_early_stop(RBM *rbm, int &epoch) {
   if (((Layer_Monitor*)monitor)->rc_monitor != NULL) {
      Reconstruction_Cost_Monitor *rc_mon = ((Layer_Monitor*)monitor)->rc_monitor;
      rc_mon->check();
#if 0
      if (rc_mon->status != OK && rc_mon->status != SLOW)
         learning = false;
      if (rc_mon->status == DONE) std::cout << "Learning converged, STOPPING" << std::endl;
      if (rc_mon->status == BROKEN) std::cout << "Learning diverging, STOPPING" << std::endl;
#elseif 0
      if (rc_mon->status == DONE) std::cout << "Learning appears to be converged.  Try stopping here" << std::endl;
#endif
      /*if (rc_mon->status == BROKEN) {
       std::cout << "Learning appears to be diverging. RESETTING" << std::endl;
       rbm->reset();
       epoch = 0;
       momentum = .5;
       k = 1;
       learning_multiplier = 75;
       }*/
      
      if (rc_mon->status == SLOW && momentum != 0 && momentum < .9) {
         momentum *= 1.05;
         std::cout << "Increasing momentum to " << momentum << std::endl;
         if (epoch > 50 && k < 5) {
            k = 5;
            for (auto connection:rbm->connections) connection->learning_rate/=100;
            std::cout << "Increasing k to " << k << std::endl;
         }
      }
   }
}