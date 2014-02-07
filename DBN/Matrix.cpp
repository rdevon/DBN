//
//  Matrix.cpp
//  DBN
//
//  Created by Devon Hjelm on 12/20/12.
//
//

#include "Matrix.h"
#include <unistd.h>
#include <math.h>
#include "SupportFunctions.h"
#include "SupportMath.h"
#include <stdlib.h>
#include <cmath>


Matrix::Matrix() : m(NULL), identity(NULL) {}

Matrix::Matrix(size_t x_size, size_t y_size, float x) :
m(x_size&&y_size ? gsl_matrix_float_alloc(x_size, y_size): NULL),
identity(x_size ? gsl_matrix_float_alloc(x_size,1) : NULL),
dim1(x_size), dim2(y_size)
{
   if (m) gsl_matrix_float_set_all(m, x);
   if (identity) gsl_matrix_float_set_all(identity, 1);
}

Matrix::Matrix(const Matrix &old) :
dim1(old.dim1), dim2(old.dim2),
m(dim2 ? gsl_matrix_float_alloc(dim1,dim2) : NULL),
identity(dim1 ? gsl_matrix_float_alloc(dim1, 1) : NULL)
{
   if (m) gsl_matrix_float_memcpy(m, old.m);
   if (identity) gsl_matrix_float_set_all(identity, 1);
}

Matrix::~Matrix() {
   if (m) gsl_matrix_float_free(m);
   if (identity) gsl_matrix_float_free(identity);
}

Matrix &Matrix::operator= (Matrix rhs) {
   swap(rhs);
   return *this;
}

float &Matrix::operator() (int i, int j) { return *gsl_matrix_float_ptr(m, i, j);}
Vector Matrix::operator() (int i) {
   Vector out(dim2);
   for (int j = 0; j < dim2; ++j) out(0,j) = (*this)(i,j);
   return out;
}


Matrix &Matrix::operator+= (const Vector &vec) {gsl_blas_sgemm(CblasNoTrans , CblasNoTrans, 1, identity, vec.m, 1, m); return *this;}
Matrix &Matrix::operator-= (const Vector &vec) {gsl_blas_sgemm(CblasNoTrans , CblasNoTrans, -1, identity, vec.m, 1, m); return *this;}
Matrix &Matrix::operator/= (const Vector &vec) {
   if (dim2 != vec.dim2) error ("Incorrect dims");
   for (int i =0; i < dim1; ++i)
      for (int j = 0; j < dim2; ++j)
         (*this)(i,j) /= *(vec.m->data + j);
   return *this;
}

Matrix &Matrix::operator+= (const Matrix &mat) {gsl_matrix_float_add(m, mat.m); return *this;}
Matrix &Matrix::operator-= (const Matrix &mat) {gsl_matrix_float_sub(m, mat.m); return *this;}
Matrix &Matrix::operator*= (const Matrix &mat) {gsl_matrix_float_mul_elements(m, mat.m); return *this;}

Matrix &Matrix::operator*= (const float &x) {gsl_matrix_float_scale(m, x); return *this;}
Matrix &Matrix::operator/= (const float &x) {gsl_matrix_float_scale(m, 1/x); return *this;}
Matrix &Matrix::operator+= (const float &x) {gsl_matrix_float_add_constant(m, x); return *this;}
Matrix &Matrix::operator-= (const float &x) {gsl_matrix_float_add_constant(m, -x); return *this;}

std::ostream& operator<<(std::ostream& out, Matrix& mat)
{
   //for(int i = 0; i < mat.dim1; ++i) {out << "--";}
   out << std::endl;
   for(int i = 0; i < mat.dim1*mat.dim2; ++i) {
      out << " " << mat.m->data[i];
      if ((i+1)%mat.dim2 == 0) out << std::endl;
   }
   //for(int i = 0; i < mat.dim1; ++i) {out << "--";}
   out << std::endl;
   return out;
}

Matrix &Matrix::times_plus(const Matrix& m1, const Matrix& m2, float a, float b, CBLAS_TRANSPOSE_t trans1, CBLAS_TRANSPOSE_t trans2) {
   gsl_blas_sgemm(trans1, trans2, a, m1.m, m2.m, b, m);
   return *this;
}

void Matrix::resize(size_t x_size, size_t y_size, float x) {
   gsl_matrix_float_free(m);
   m = gsl_matrix_float_alloc(x_size, y_size);
   gsl_matrix_float_set_all(m, x);
}

void Matrix::dropout(float p) {
   apply([&](float &x) {x*=((gsl_rng_uniform(r))>=p);});
}

void Matrix::sample() {
   apply([&](float &x) {x =(gsl_rng_uniform(r) < x);});
}

void Matrix::add_gaussian_noise(float sigma) {
   apply([&](float &x) {x+= gsl_ran_gaussian(r, sigma);});
}

void Matrix::add_relu_noise(const Matrix &expectations) {
   if (dim1 != expectations.dim1 || dim2 != expectations.dim2) error("Incorrect dims");
   for (int i = 0; i < dim1*dim2; ++i) {
      *(m->data + i) += gsl_ran_gaussian(r, sigmoid(*(expectations.m->data +i)));
      if (*(m->data + i) < 0) *(m->data + i) = 0;
   }
}

void Matrix::norm_rows() {
   for (int i = 0; i < dim1; ++i) {
      float agg = 0;
      for (int j = 0; j < dim2; ++j) agg += *(m->data + dim2*i + j);
#if DEBUG
      if (agg == 0) {
         std::cout << *this;
         error("Out of precision");
      }
#endif
      if (agg != 0) for (int j = 0; j < dim2; ++j) *(m->data + dim2*i + j) /= agg;
   }
}

void Matrix::row_pick_top() {
   for (int i = 0; i < dim1; ++i) {
      float max = 0;
      int max_index;
      for (int j = 0; j < dim2; ++j) {
         float val = *(m->data+dim2*i + j);
         if (val > max) {
            max_index = j;
            max = val;
         }
         *(m->data+dim2*i + j) = 0;
      }
      *(m->data + dim2*i + max_index) = 1;
   }
}

void Matrix::copy_submatrices(const Matrix &from, int beg_index) {
   if (beg_index + dim1 > from.dim1) error("Copying incorrect sizes");
   for (int i = 0; i < dim1*dim2; ++i) {
      *(m->data+i) = *(from.m->data+beg_index*dim2+i);
   }
}

void Matrix::fill_submatrix(const Matrix &from, int beg_index) {
   if (dim2 != from.dim2 || beg_index+from.dim1 > dim1) error("Filling incorrect sizes");
   for (int i = 0; i < from.dim1*from.dim2; ++i) {
      *(m->data+beg_index*dim2 + i) = *(from.m->data + i);
   }
}

void Matrix::fill_submatrix(const Matrix &from, int beg_c_index, int beg_r_index) {
   if (beg_c_index+ from.dim1 > dim1 || beg_r_index + from.dim2 > dim2) error("Filling incorrect sizes");
   for (int i = 0; i < from.dim1*from.dim2; ++i) {
      *(m->data + beg_c_index*dim2 + i + beg_r_index) = *(from.m->data + i);
   }
}

Vector Matrix::mean_image() {
   Vector mean_image(dim2);
   for (int i = 0; i < dim1; ++i) {
      for (int j = 0; j < dim2; ++j) {
         mean_image(0,j) += (*this)(i,j);
      }
   }
   mean_image /= dim1;
   return mean_image;
}

Vector Matrix::sd_image() {
   Vector mi = mean_image();
   Vector sd_image(dim2);
   for (int i = 0; i < dim1; ++i) {
      for (int j = 0; j < dim2; ++j) {
         sd_image(0,j) += powf(mi(0,j)-(*this)(i,j), 2);
      }
   }
   sd_image /= dim1;
   sd_image.apply([](float &x){x = sqrtf(x);});
   return sd_image;
}

float Matrix::mean() {return gsl_stats_float_mean(m->data, 1, dim1*dim2);}

void Matrix::normalize() {
   float me = gsl_stats_float_mean(m->data, 1, dim1*dim2);
   float sd = gsl_stats_float_sd(m->data, 1, dim1*dim2);
   (*this)-=me;
   (*this)/=sd;
}

void Matrix::set_all(float x) { gsl_matrix_float_set_all(m, x); }

void Matrix::set_gaussian(float sigma) { for (int i = 0; i < m->size1*m->size2; ++i) *(m->data+i) = gsl_ran_gaussian(r, sigma); }

float cross_validate(const Matrix& mat1, const Matrix &mat2) {
   if (mat1.dim1 != mat2.dim1 || mat1.dim2 != mat2.dim2) error("Cross Validation with Mismatched dimensions");
   float cv = 0;
   for (int i = 0; i < mat1.dim1*mat2.dim2; ++i) {
      float val1 = *(mat1.m->data + i);
      float val2 = *(mat2.m->data + i);
      cv -= val1 * logf(val2) + (1-val1) * logf(1-val2);
      //std::cout << val1 << " " << val2 << std::endl;
   }
   return cv;
}

void Matrix::min_max(float &min, float &max) { gsl_matrix_float_minmax(m, &min, &max); }
float Matrix::max() {return gsl_matrix_float_max(m);}

void Matrix::row_zeromean_unitvar() {
   for (int i = 0; i < dim1; ++i) {
      float mean = 0, var = 0;
      for (int j = 0; j < dim2; ++j) mean += *(m->data + dim2*i + j);
      mean /= dim2;
      for (int j = 0; j < dim2; ++j) var += powf(*(m->data + dim2*i + j)-mean,2);
      var /= (dim2-1);
      for (int j = 0; j < dim2; ++j) {
         (*(m->data + dim2*i + j)) -= mean;
         (*(m->data + dim2*i + j)) /= sqrtf(var);
      }
   }
}

void Matrix::flatten_rows(float row_length) {
   for (int i = 0; i < dim1; ++i) {
      double length = 0;
      for (int j = 0; j < dim2; ++j) length += pow(*(m->data + dim2*i + j),2);
      if (sqrt(length) > row_length) for (int j = 0; j < dim2; ++j) *(m->data + dim2*i + j) *= (row_length/sqrt(length));
   }
}

void Matrix::flatten_columns(float column_length) {
   for (int j = 0; j < dim2; ++j) {
      float length = 0;
      for (int i = 0; i < dim1; ++i) length += powf(*(m->data + dim2*i + j), 2);
      if (sqrt(length) > column_length) for (int i = 0; i < dim1; ++i) *(m->data + dim2*i + j) *= (column_length/sqrt(length));
   }
}

void Matrix::adjust_gain(const Matrix &prev, const Matrix &curr) {
   for (int i = 0; i < dim1*dim2; ++i) {
      float pval = *(prev.m->data+i);
      float cval = *(curr.m->data+i);
      if (pval * cval > 0) *(m->data+i) += 0.05;
      else *(m->data+i) *=.95;
      if (*(m->data+i) < 0.001) *(m->data+i) = 0.001;
      if (*(m->data+i) > 100) *(m->data+i) = 100;
   }
}

void Matrix::remove_mask(const Vector& mask) {
   int count = (int)std::count(mask.m->data, (mask.m->data) + mask.dim2, 1);
   Matrix newmat(dim1, count);
   
   for (int i = 0; i < dim1; ++i) {
      for (int j = 0, jprime = 0; j < dim2; ++j) {
         if (*(mask.m->data + j) == 1) {
            newmat(i,jprime) = (*this)(i,j);
            ++jprime;
         }
      }
   }
   (*this) = newmat;
}

void Matrix::load(const Vector& vec, int index) {
   for (int i = 0; i < dim1*dim2; ++i) {
      *(m->data + i) = *(vec.m->data + dim1*dim2*index + i);
   }
}

void Matrix::shuffle_rows(gsl_rng *rand) {
   gsl_ran_shuffle(rand, m->data, dim1, dim2*sizeof(float));
}

Matrix transpose(const Matrix& other) {
   Matrix t_mat(other.dim2, other.dim1);
   gsl_matrix_float_transpose_memcpy(t_mat.m, other.m);
   return t_mat;
}

//------ Vector Class

Vector::Vector() : Matrix() {}
Vector::~Vector() {}

Vector::Vector(size_t x_size, float x) : Matrix(1, x_size,x) {}

Vector &Vector::operator= (Vector rhs) {
   swap(rhs);
   return *this;
}

Vector Vector::make_mask() {
   Vector mask(dim2);
   float me = mean();
   for (int i = 0; i < dim2; ++i) {
      if (*(m->data + i) > me) *(mask.m->data + i) = 1;
   }
   return mask;
}

Vector Vector::add_mask(const Vector &mask) {
   Vector newvec(mask.dim2);
   for (int i = 0, iprime = 0; i < mask.dim2; ++i) {
      if (*(mask.m->data + i) == 1) {
         newvec(0,i) = (*this)(0,iprime);
         ++iprime;
      }
      else newvec(0,i) = WHITE;
   }
   return newvec;
}

void Matrix::catch_nan_or_inf() {
   bool nanorinf = false;
   for (int i = 0; i < dim1*dim2; ++i) {
      float &val = *(m->data + i);
      if (std::isnan(val) || std::isinf(val)) {
         nanorinf = true;
         break;
      }
   }
   if (nanorinf) {
      std::cout << *this;
      error("NaN or Inf");
   }
}

float distance(const Matrix& mat1, const Matrix& mat2) {
   float distance = 0;
   for (int i = 0; i < mat1.dim1*mat1.dim2; ++i) {
      float m1 = *(mat1.m->data + i);
      float m2 = *(mat2.m->data + i);
      distance += powf(m1-m2,2);
   }
   distance /= mat1.dim1;
   return distance;
}

void Matrix::save(std::string filename) {
   std::ofstream outstream;
   outstream.open(filename.c_str());
   outstream << *this;
   outstream.close();
}

Matrix Matrix::guess_classes() {
   Matrix guess = *this;
   guess.apply([&](float &x) {x = floor(x+0.5);});
   return guess;
}
