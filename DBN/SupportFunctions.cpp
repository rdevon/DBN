//
//  SupportFunctions.cpp
//  DBN
//
//  Created by Devon Hjelm on 7/20/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "SupportFunctions.h"

std::string readTextFile(const std::string& filename)
{
   std::ifstream infile(filename.c_str()); // File stream
   std::string source;                     // Text file string
   std::string line;                       // A line in the file
   
   // Make sure the file could be opened
   if(!infile.is_open())
   {
      std::cerr << "Could not open file: " << filename << std::endl;
   }
   
   // Read in the source one line at a time, then append it
   // to the source string. Not efficient.
   while(infile.good())
   {
      getline(infile, line);
      source = source + line + "\n";
   }
   
   infile.close();
   return source;
}

void error(std::string message) {
   std::cout << "ERROR: " << message << std::endl;
   exit(EXIT_FAILURE);
}

std::vector<float> load_dlm(std::string filename) {
   std::vector<float> values;
   std::ifstream file;
   file.open(filename.c_str());
   float value;
   std::string line;
   getline(file, line);
   std::istringstream iss(line);
   while ( iss >> value) values.push_back(value);
   return values;
}

std::ostream& operator<<(std::ostream& out, std::vector<float> vec) {
   for (auto val:vec) out << val << " ";
   return out;
}

void save(std::vector<float> vec, std::string file_name) {
   std::ofstream outstream;
   outstream.open(file_name.c_str());
   outstream << vec;
   outstream.close();
}