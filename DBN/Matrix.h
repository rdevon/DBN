//
//  Matrix.h
//  DBN
//
//  Created by Devon Hjelm on 12/9/12.
//
//

#ifndef DBN_Matrix_h
#define DBN_Matrix_h

#include <iostream>
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_vector.h"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_permute.h>
#include "gsl/gsl_statistics.h"

class Vector;

extern gsl_rng *r;

class Matrix {
public:
   size_t dim1,dim2;
   
   //------ Rule of Three
   
   Matrix();
   Matrix(const Matrix &old);
   ~Matrix();
   Matrix(size_t x_size, size_t y_size, float x=0);
   
   void swap(Matrix &other) {
      using std::swap;
      swap(dim1, other.dim1);
      swap(dim2, other.dim2);
      swap(m, other.m);
      swap(identity, other.identity);
   }
   
   Matrix &operator= (Matrix mat);
   
   //------
   
   void resize(size_t x_size, size_t y_size, float x);
   
   void set_all(float x);
   
   template <class Function>
   void set_all(Function fun) { for (int i = 0; i < dim1*dim2; ++i) *(m->data+i) = fun; }
   
   void set_gaussian(float sigma = 1);
   
   Matrix &operator*= (const float &x);
   Matrix &operator/= (const float &x);
   Matrix &operator+= (const float &x);
   Matrix &operator-= (const float &x);
   
   Matrix &operator*= (const Matrix &mat);
   Matrix &operator+= (const Matrix &mat);
   Matrix &operator-= (const Matrix &mat);
   Matrix &operator+= (const Vector &vec);
   Matrix &operator-= (const Vector &vec);
   Matrix &operator/= (const Vector &vec);
   
   float &operator() (int i, int j);
   Vector operator() (int i);
   
   Matrix &times_plus(const Matrix&, const Matrix&, float a, float b, CBLAS_TRANSPOSE_t, CBLAS_TRANSPOSE_t);
   friend Matrix transpose(const Matrix &);
   
   template <class Function>
   void apply(Function fun) {
      for (int i = 0; i < dim1*dim2; ++i) {
         float *v = m->data+i;
         fun(*v);
      }
   }
   
   void copy_submatrices(const Matrix &from, int beg_index);
   void fill_submatrix(const Matrix &from, int beg_index);
   void fill_submatrix(const Matrix &from, int beg_c_index, int beg_r_index);
   void min_max(float &, float&);
   Vector mean_image();
   Vector sd_image();
   float mean();
   void normalize();
   
   //------ Special functions needed for DBN
   
   void catch_nan_or_inf();
   void dropout(float p);
   void sample();
   void add_gaussian_noise(float sigma =1);
   void add_relu_noise(const Matrix&);
   void norm_rows();
   void row_pick_top();
   void row_zeromean_unitvar();
   void flatten_rows(float row_length);
   void flatten_columns(float column_length);
   void adjust_gain(const Matrix &prev, const Matrix &curr);
   void remove_mask(const Vector& mask);
   void load(const Vector&, int i = 0);
   void shuffle_rows(gsl_rng *rand);
   float max();
   void save(std::string filename);
   Matrix guess_classes();
   
   friend float cross_validate(const Matrix& mat1, const Matrix& mat2);
   
   gsl_matrix_float *m;
   gsl_matrix_float *identity;
};

std::ostream& operator<<(std::ostream& out, Matrix& mat);

class Vector : public Matrix {
public:
   
   ~Vector();
   Vector();
   Vector(size_t x_size, float x=0);
   Vector &operator= (Vector);
   //Vector &operator-= (const Vector &vec);
   
   void swap(Vector &other) {
      using std::swap;
      swap(dim1, other.dim1);
      swap(dim2, other.dim2);
      swap(m, other.m);
   }
   
   Vector make_mask();
   Vector add_mask(const Vector &mask);
};

inline Matrix operator* (const float &lhs, Matrix rhs) {
   rhs*=lhs;
   return rhs;
}

inline Matrix operator* (Matrix &lhs, const float &rhs) {return rhs*lhs;}

inline Matrix operator+ (Matrix lhs, const Matrix &rhs) {
   lhs+=rhs;
   return lhs;
}

inline Matrix operator- (Matrix lhs, const Matrix &rhs) {
   lhs-=rhs;
   return lhs;
}

inline Matrix operator* (Matrix mat1, Matrix &mat2) {
   mat1*=mat2;
   return mat1;
}

float distance(const Matrix& mat1, const Matrix& mat2);

#endif