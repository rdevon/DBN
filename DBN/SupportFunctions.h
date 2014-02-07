//
//  SupportFunctions.h
//  DBN
//
//  Created by Devon Hjelm on 7/20/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef DBN_SupportFunctions_h
#define DBN_SupportFunctions_h
#include "Params.h"
std::string readTextFile(const std::string& filename);

template<typename T>
std::string convert_to_string(T value) {
   std::stringstream out;
   out << value;
   return out.str();
}

template<typename T>
bool x_is_in(T x, std::vector<T> vec) {
   return (std::find(vec.begin(), vec.end(), x) != vec.end());
}

template<typename T>
int get_index(T x, std::vector<T> vec) {
   auto iter = std::find_if(vec.begin(), vec.end(), [&](T y){return x == y;});
   return (int)std::distance(vec.begin(), iter);
}

void error(std::string message);

std::vector<float> load_dlm(std::string filename);

void save(std::vector<float>, std::string filename);

std::ostream& operator<<(std::ostream& out, std::vector<float>);

#endif
