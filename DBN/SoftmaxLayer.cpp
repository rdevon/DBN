//
//  CSoftmaxLayer.cpp
//  DBN
//
//  Created by Devon Hjelm on 8/7/12.
//
//

#include "Layers.h"
#include "IO.h"

void SoftmaxLayer::getExpectations(){
   //Apply continuous softmax.
   for (int i = 0; i < nodenum; ++i){
      for (int j = 0; j < batchsize; ++j){
         float denom = 0;
         for (int c = 0; c < nodenum; ++c) {
            float c_val = gsl_matrix_float_get(activations, c, j);
            denom += expf(c_val);
         }
         float num = gsl_matrix_float_get(activations, i, j);
         gsl_matrix_float_set(expectations, i, j, num/denom);
      }
   }
}

void SoftmaxLayer::sample(){
   for (int i = 0; i < nodenum; ++i){
      for (int j = 0; j < batchsize; ++j){
         float u = gsl_rng_uniform(r);
         float sample = (float)(gsl_matrix_float_get(expectations, i, j) > u);
         gsl_matrix_float_set(samples, i, j, sample);
      }
   }
}

void SoftmaxLayer::update(ContrastiveDivergence *teacher){
   Layer::update(teacher);
}

float SoftmaxLayer::reconstructionCost(gsl_matrix_float *dataMat, gsl_matrix_float *modelMat){
   reconstruction_cost=0;
   gsl_matrix_float *squared_error = gsl_matrix_float_alloc(dataMat->size1, dataMat->size2);
   gsl_matrix_float_memcpy(squared_error, dataMat);
   gsl_matrix_float_sub(squared_error, modelMat);
   gsl_matrix_float_mul_elements(squared_error, squared_error);
   for (int i = 0; i < squared_error->size1; ++i)
      for (int j = 0; j < squared_error->size2; ++j)
         reconstruction_cost+=gsl_matrix_float_get(squared_error, i, j);
   reconstruction_cost /= squared_error->size2;
   gsl_matrix_float_free(squared_error);
   return reconstruction_cost;
}

float SoftmaxLayer :: freeEnergy_contibution() {return 0;}

//The input needs to be shaped depending on the type of visible layer.
void SoftmaxLayer::shapeInput(DataSet *data){
}

