//
//  MNIST.cpp
//  DBN
//
//  Created by Devon Hjelm on 11/6/12.
//
//

#include "SupportFunctions.h"
#include "DataSets.h"

DataSet *load_MNIST_DS(std::string pathname)
{
   std::cout << "Loading MNIST" << std::endl;
   
   std::string trainfile = pathname + "train-images.idx3-ubyte";
   std::string testfile = pathname + "t10k-images.idx3-ubyte";
   FILE *file_handle;
   file_handle = fopen(trainfile.c_str(), "rb");
   
   unsigned char pixel;
   uint32_t magicnumber, imageNum, rowNum, colNum;
   
   fread(&magicnumber, sizeof(magicnumber), 1, file_handle);
   fread(&imageNum, sizeof(imageNum), 1, file_handle);
   fread(&rowNum, sizeof(rowNum), 1, file_handle);
   fread(&colNum, sizeof(colNum), 1, file_handle);
   
   magicnumber = ntohl(magicnumber);
   imageNum = ntohl(imageNum);
   rowNum = ntohl(rowNum);
   colNum = ntohl(colNum);
   
   DataSet *dataset = new DataSet(MNIST, imageNum, rowNum, colNum, 1);
   dataset->data_path = pathname;
   
   for (int i = 0; i < imageNum; ++i)
      for (int j = 0; j < rowNum*colNum; ++j){
         fread(&pixel, sizeof(pixel), 1, file_handle);
         dataset->data(i,j) = pixel;
      }
   fclose(file_handle);
#if 0
   file_handle = fopen(testfile.c_str(), "rb");
   
   fread(&magicnumber, sizeof(magicnumber), 1, file_handle);
   fread(&imageNum, sizeof(imageNum), 1, file_handle);
   fread(&rowNum, sizeof(rowNum), 1, file_handle);
   fread(&colNum, sizeof(colNum), 1, file_handle);
   
   magicnumber = ntohl(magicnumber);
   imageNum = ntohl(imageNum);
   rowNum = ntohl(rowNum);
   colNum = ntohl(colNum);

   new (&test) Matrix(imageNum, colNum*rowNum);
   
   for (int i = 0; i < imageNum; ++i)
      for (int j = 0; j < rowNum*colNum; ++j){
         fread(&test(i,j), sizeof(pixel), 1, file_handle);
      }
   
   fclose(file_handle);
#endif
   
   float min,max;
   dataset->data.min_max(min, max);
   dataset->data-=min;
   dataset->data/=(max-min);
   std::cout << "Done loading" << std::endl;
   return dataset;
}

DataSet *load_MNIST_L(std::string pathname){
   
   std::cout << "Loading MNIST labels" << std::endl;
   
   std::string trainfile = MNISTpath + "train-labels.idx1-ubyte";
   FILE *file_handle;
   file_handle = fopen(trainfile.c_str(), "rb");
   
   unsigned char label;
   uint32_t magicnumber, imageNum;
   
   fread(&magicnumber, sizeof(magicnumber), 1, file_handle);
   fread(&imageNum, sizeof(imageNum), 1, file_handle);
   
   magicnumber = ntohl(magicnumber);
   imageNum = ntohl(imageNum);
   
   DataSet *dataset = new DataSet(MNIST_L, imageNum, 10, 1, 1);
   dataset->data_path = pathname;
   
   for (int i = 0; i < imageNum; ++i) {
      fread(&label, sizeof(label), 1, file_handle);
      dataset->data(i,label) = 1;
   }
   
   fclose(file_handle);
   std::cout << "Done loading" << std::endl;
   return dataset;
}