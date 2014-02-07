//
//  GaussianLayer2.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/6/12.
//
//

#include <iostream>
#include "Layers.h"
#include "DataSets.h"

void GaussianLayer::get_expectations(Purpose_t purpose)
{}

void GaussianLayer::sample() {m_learning.add_gaussian_noise();}

void GaussianLayer::get_derivatives() {
   gradient = m_learning;
   gradient.set_all(1);
}

void GaussianLayer::shapeInput(DataSet *data){
   Matrix input = data->train;
   input.row_zeromean_unitvar();
}
