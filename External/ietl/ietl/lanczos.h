/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 2001-2011 by Prakash Dayal <prakash@comp-phys.org>,
 *                            Matthias Troyer <troyer@comp-phys.org>
 *
 * This software is part of the ALPS libraries, published under the ALPS
 * Library License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 *
 * You should have received a copy of the ALPS Library License along with
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

/* $Id: lanczos.h,v 1.34 2004/06/29 08:31:02 troyer Exp $ */
/* Modified by Alexander Wietek, 2018/10/22 */

#ifndef EXTERNAL_IETL_IETL_LANCZOS_H_
#define EXTERNAL_IETL_IETL_LANCZOS_H_

#include <ietl/complex.h>
#include <ietl/eigeninterface.h>
#include <ietl/fmatrix.h>
#include <ietl/ietl2lapack.h>
#include <ietl/iteration.h>
#include <ietl/tmatrix.h>
#include <climits>
#include <exception>
#include <stdexcept>
#include <vector>

namespace ietl {

// class Info starts, contains info about the eigen vector calculation.
template <class magnitude_type = double>
class Info {
 public:
  enum errorinfo { ok = 0, no_eigenvalue, not_calculated };
  Info() {}
  Info(std::vector<int> M1, std::vector<int> M2, std::vector<int> Ma,
       std::vector<magnitude_type> Eigenvalue,
       std::vector<magnitude_type> Residuum, std::vector<errorinfo> Status)
      : m1_(M1),
        m2_(M2),
        ma_(Ma),
        eigenvalue_(Eigenvalue),
        residuum_(Residuum),
        status_(Status) {}

  int m1(int i) const { return m1_[i]; }
  int m2(int i) const { return m2_[i]; }
  int ma(int i) const { return ma_[i]; }
  int size() { return m1_.size(); }
  magnitude_type eigenvalue(int i) const { return eigenvalue_[i]; }
  magnitude_type residual(int i) const { return residuum_[i]; }
  errorinfo error_info(int i) const { return status_[i]; }

 private:
  std::vector<int> m1_;
  std::vector<int> m2_;
  std::vector<int> ma_;
  std::vector<magnitude_type> eigenvalue_;
  std::vector<magnitude_type> residuum_;
  std::vector<errorinfo> status_;
};  // end of class Info.

// class lanczos starts:
template <class MATRIX, class VS>
class lanczos : public Tmatrix<VS> {
  typedef Tmatrix<VS> super_type;

 public:
  typedef typename vectorspace_traits<VS>::vector_type vector_type;
  typedef typename vectorspace_traits<VS>::scalar_type scalar_type;
  typedef typename vectorspace_traits<VS>::magnitude_type magnitude_type;

  lanczos(const MATRIX& matrix, const VS& vec);  // constructor.

  template <class IT, class GEN>
  void calculate_eigenvalues(IT& iter, GEN gen) {
    generate_tmatrix(iter, gen);
  }

  template <class IT>
  void more_eigenvalues(IT& iter) {
    generate_tmatrix(iter);
  }

  template <class IN, class OUT, class GEN>
  void eigenvectors(IN in_eigvals_start, IN in_eigvals_end, OUT eig_vectors,
                    Info<magnitude_type>& inf, GEN gen, int maxiter = 0,
                    int maxcount = 50);

  std::vector<std::vector<magnitude_type> > const& t_eigenvectors() {
    return Tvectors;
  }

  template <class Archive>
  void save(Archive& ar) const {
    ar << static_cast<super_type const&>(*this);
    ar << startvector << vec2 << n << M1 << M2 << Ma << Tvectors;
  }

  template <class Archive>
  void load(Archive& ar) {
    ar >> static_cast<super_type&>(*this);
    ar >> startvector >> vec2 >> n >> M1 >> M2 >> Ma >> Tvectors;
  }

 private:
  template <class IN>
  void find_m1m2(IN in_eigvals_start, IN in_eigvals_end);
  // m1 m2 finder for eigen vector calculation.

  template <class GEN>
  std::pair<magnitude_type, magnitude_type> make_first_step(GEN gen);
  std::pair<magnitude_type, magnitude_type> make_step(int j, vector_type& vec3);

  template <class IT, class GEN>
  void generate_tmatrix(IT& iter, GEN gen);
  template <class IT>
  void generate_tmatrix(IT& iter);
  // T matrix generator, Used in eigenvalues & eigenvectors calculation.

  const MATRIX& matrix_;
  const VS vecspace_;
  vector_type startvector;
  vector_type vec2;
  unsigned int n;  // index of vec2
  std::vector<int> M1, M2, Ma;
  std::vector<std::vector<magnitude_type> >
      Tvectors;  // contains eigenvectors of T matrix.

};  // end of class lanczos.

//-----------------------------------------------------------------------
// implementation of member functions start:
template <class MATRIX, class VS>  // constructor:
lanczos<MATRIX, VS>::lanczos(const MATRIX& matrix, const VS& vec)
    : matrix_(matrix),
      vecspace_(vec),
      startvector(new_vector(vec)),
      vec2(new_vector(vec)),
      n(0) {}

//-----------------------------------------------------------------------
// eigen vectors calculation starts:
template <class MATRIX, class VS>
template <class IN, class OUT, class GEN>
void lanczos<MATRIX, VS>::eigenvectors(IN in_eigvals_start, IN in_eigvals_end,
                                       OUT eig_vectors,
                                       Info<magnitude_type>& inf, GEN gen_,
                                       int maxiter, int maxcount) {
  vector_type vec3 = new_vector(vecspace_);  // a temporary vector.
  Tvectors.resize(0);
  std::vector<vector_type> eigvectors;  // contains ritz vectors.
  // calculation of eigen vectors of T matrix(consists of alphas & betas):
  int n1 = super_type::alpha.size();
  magnitude_type mamax, error, lambda;
  std::pair<magnitude_type, magnitude_type> a_and_b;
  unsigned int ma = 0, deltam;
  int nth, count;
  unsigned int maMax = 0;
  std::vector<magnitude_type> eigenval_a, residuum;
  std::vector<typename Info<magnitude_type>::errorinfo> status;

  find_m1m2(in_eigvals_start, in_eigvals_end);
  std::vector<int>::iterator M1_itr = M1.begin();
  std::vector<int>::iterator M2_itr = M2.begin();

  while (in_eigvals_start != in_eigvals_end) {
    lambda = 0;
    count = 0;
    typename Info<magnitude_type>::errorinfo errInf = Info<magnitude_type>::ok;

    // calculation of ma starts:
    if (*M1_itr != 0 && *M2_itr != 0) {
      ma = (3 * (*M1_itr) + *M2_itr) / 4 + 1;
      deltam = ((3 * (*M1_itr) + 5 * (*M2_itr)) / 8 + 1 - ma) / 10 + 1;
    } else if (*M1_itr != 0 && *M2_itr == 0) {
      ma = (5 * (*M1_itr)) / 4 + 1;
      mamax = std::min((11 * n1) / 8 + 12, (13 * (*M1_itr)) / 8 + 1);
      deltam = int((mamax - ma) / 10) + 1;
      if (maxiter > 0) maxcount = maxiter / deltam;
    } else {
      errInf = Info<magnitude_type>::no_eigenvalue;
      ma = 0;
    }  // calculation of ma ends.
    eigvectors.push_back(new_vector(vecspace_));
    // new ritz vector is being added in eigvectors.

    std::vector<magnitude_type> Tvector;
    Tvectors.push_back(Tvector);
    // new T matrix vector is being added in Tvectors.

    if (ma == 0) eigvectors.back() *= 0.;

    if (ma != 0) {
      std::vector<magnitude_type> eval(ma);
      ietl::FortranMatrix<magnitude_type> z(ma, ma);
      // on return, z contains all orthonormal eigen vectors of T matrix.
      do {
        if (ma >
            super_type::alpha.size()) {  // size of T matrix is to be increased.
          ietl::fixed_lanczos_iteration<magnitude_type> iter_temp(ma);
          generate_tmatrix(iter_temp, gen_);
        }
        count++;
        int info =
            ietl2lapack::stev(super_type::alpha, super_type::beta, eval, z, ma);
        if (info > 0)
          throw std::runtime_error("LAPACK error, stev function failed.");

        // search for the value of nth starts, where nth is the nth eigen vector
        // in z.
        for (nth = ma - 1; nth >= 0; nth--)
          if (fabs(eval[nth] - *in_eigvals_start) <= super_type::thold) break;

        // search for the value of ith ends, where ith is the ith eigen vector
        // in z.
        if (nth == -1) {
          error = 0;
          ma = 0;
          eigvectors.back() *= 0.;
          errInf = Info<magnitude_type>::no_eigenvalue;
        } else {
          error = fabs(super_type::beta[ma - 1] *
                       z(ma - 1, nth));  // beta[ma - 1] = betaMplusOne.
          if (error > super_type::error_tol) {
            ma += deltam;
            eval.resize(ma);
            z.resize(ma, ma);
          }
        }  // end of else
      } while (error > super_type::error_tol && count < maxcount);

      if (error > super_type::error_tol) {
        eigvectors.back() *= 0.;
        errInf = Info<magnitude_type>::not_calculated;
      } else {  // if error is small enough.
        if (ma != 0) {
          for (unsigned int i = 0; i < ma; i++)
            (Tvectors.back()).push_back(z(i, nth));
          if (ma > maMax) maMax = ma;
          lambda = eval[nth];
        }  // end of if(ma != 0), inner.
      }    // end of else{//if error is small enough.
    }      // end of if(ma != 0).

    eigenval_a.push_back(lambda);  // for Info object.
    Ma.push_back(ma);              // for Info object.
    status.push_back(errInf);
    in_eigvals_start++;
    M1_itr++;
    M2_itr++;
  }  // end of while(in_eigvals_start !=  in_eigvals_end)

  // basis transformation of eigen vectors of T. These vectors are good
  // approximation of eigen vectors of actual matrix.
  typename std::vector<vector_type>::iterator eigenvectors_itr;
  typename std::vector<std::vector<magnitude_type> >::iterator Tvectors_itr;

  a_and_b = make_first_step(gen_);
  if (a_and_b.first != super_type::alpha[0] ||
      a_and_b.second != super_type::beta[0])
    throw std::runtime_error("T-matrix problem at first step");

  eigenvectors_itr = eigvectors.begin();
  Tvectors_itr = Tvectors.begin();

  while (eigenvectors_itr != eigvectors.end()) {
    if (!Tvectors_itr->empty()) {
      *eigenvectors_itr = (*Tvectors_itr)[0] * startvector;
      *eigenvectors_itr += (*Tvectors_itr)[1] * vec2;
    }
    eigenvectors_itr++;
    Tvectors_itr++;
  }
  n = 2;
  for (unsigned int j = 2; j < maMax; j++) {
    a_and_b = make_step(j - 1, vec3);
    if (a_and_b.first != super_type::alpha[j - 1] ||
        a_and_b.second != super_type::beta[j - 1])
      throw std::runtime_error("T-matrix problem");

    ++n;
    eigenvectors_itr = eigvectors.begin();
    Tvectors_itr = Tvectors.begin();
    while (eigenvectors_itr != eigvectors.end()) {
      if (Tvectors_itr->size() > j &&
          std::abs((*Tvectors_itr)[j]) > super_type::error_tol) {
        *eigenvectors_itr += (*Tvectors_itr)[j] * vec2;
        // vec2 is being added in one vector of eigvectors.
      }
      eigenvectors_itr++;
      Tvectors_itr++;
    }  // end of while loop.
  }    // end of for(int j = 2; j < maMax; j++).
       // end of basis transformation.

  // copying to the output iterator & residuum calculation starts:
  int i = 0;
  for (eigenvectors_itr = eigvectors.begin();
       eigenvectors_itr != eigvectors.end(); eigenvectors_itr++) {
    *eig_vectors = *eigenvectors_itr;
    eig_vectors++;
    ietl::mult(matrix_, *eigenvectors_itr, vec3);
    vec3 -= eigenval_a[i++] * (*eigenvectors_itr);

    // now vec3 is (A*v - eigenval_a*v); *eigenvectors_itr) is being added in
    // vec3.
    residuum.push_back(ietl::two_norm(vec3));
  }  // copying to the output iterator ends.
  inf = Info<magnitude_type>(M1, M2, Ma, eigenval_a, residuum, status);
}  // end of void eigenvector(....).

//------------------------------------------------------
template <class MATRIX, class VS>
template <class IN>
void lanczos<MATRIX, VS>::find_m1m2(IN in_eigvals_start, IN in_eigvals_end) {
  int info, m2counter = 0;
  // unsigned int n = 1;
  n = 1;
  IN in_eigvals = in_eigvals_start;
  M1.resize(in_eigvals_end - in_eigvals_start, 0);
  M2.resize(in_eigvals_end - in_eigvals_start, 0);

  while (m2counter < (in_eigvals_end - in_eigvals_start) &&
         (n < super_type::alpha.size())) {
    n++;  // n++ == 2, at first time in this loop.
    std::vector<magnitude_type> eval(n);
    info = ietl2lapack::stev(super_type::alpha, super_type::beta, eval, n);
    if (info > 0)
      throw std::runtime_error("LAPACK error, stev function failed.");

    std::vector<int>::iterator M1_itr = M1.begin();
    std::vector<int>::iterator M2_itr = M2.begin();
    in_eigvals = in_eigvals_start;

    while (in_eigvals != in_eigvals_end) {
      if (*M1_itr == 0 || *M2_itr == 0) {
        typename std::vector<magnitude_type>::const_iterator lb, ub;
        ub = std::lower_bound(eval.begin(), eval.end(),
                              *in_eigvals + super_type::thold);
        lb = std::upper_bound(eval.begin(), eval.end(),
                              *in_eigvals - super_type::thold);
        if (*M1_itr == 0 && ub - lb >= 1) *M1_itr = n;
        if (*M2_itr == 0 && ub - lb >= 2) {
          *M2_itr = n;
          m2counter++;
        }
      }  // end of "if(*M1_itr == 0 || *M2_itr ...".
      in_eigvals++;
      M1_itr++;
      M2_itr++;
    }  // end of inner while loop.
  }    // end of outer while loop.
}  // end of function find_m1m2.

//------------------------------------------------------
// generation of alpha, beta of T matrix starts:
template <class MATRIX, class VS>
template <class IT, class GEN>
void lanczos<MATRIX, VS>::generate_tmatrix(IT& iter, GEN gen_) {
  vector_type vec3 = new_vector(vecspace_);
  std::pair<magnitude_type, magnitude_type> a_and_b;

  if (super_type::alpha.size() == 0) {
    a_and_b = make_first_step(gen_);
    Tmatrix<VS>::push_back(a_and_b);  // member of T-matrix class.
    n = 1;
  }

  generate_tmatrix(iter);
}  // generation of alpha, beta of T matrix ends.

template <class MATRIX, class VS>
template <class IT>
void lanczos<MATRIX, VS>::generate_tmatrix(IT& iter) {
  vector_type vec3 = new_vector(vecspace_);
  std::pair<magnitude_type, magnitude_type> a_and_b;

  for (unsigned int j = 0; j < n; j++) ++iter;
  if (super_type::alpha.size() == 0)
    throw std::runtime_error(
        "T matrix error, size of T matrix is zero, more_eigenvalues() cannot "
        "be called.");

  while (!iter.finished(*this)) {
    a_and_b = make_step(n, vec3);
    if (n == super_type::alpha.size())
      Tmatrix<VS>::push_back(a_and_b);  // member of T-matrix class
    ++iter;
    ++n;
  }
}  // generation of alpha, beta of T matrix ends.

//------------------------------------------------------
// generation of one step of alpha, beta:
template <class MATRIX, class VS>
template <class GEN>
std::pair<typename lanczos<MATRIX, VS>::magnitude_type,
          typename lanczos<MATRIX, VS>::magnitude_type>
lanczos<MATRIX, VS>::make_first_step(GEN gen) {
  magnitude_type a, b;
  ietl::generate(startvector, gen);
  ietl::project(startvector, vecspace_);
  startvector /= ietl::two_norm(startvector);  // normalization of startvector.
  ietl::mult(matrix_, startvector, vec2);
  a = ietl::real(ietl::dot(startvector, vec2));
  vec2 -= a * startvector;
  b = ietl::two_norm(vec2);
  vec2 /= b;
  return std::make_pair(a, b);
}

template <class MATRIX, class VS>
std::pair<typename lanczos<MATRIX, VS>::magnitude_type,
          typename lanczos<MATRIX, VS>::magnitude_type>
lanczos<MATRIX, VS>::make_step(int j, vector_type& vec3) {
  magnitude_type a, b;
  b = super_type::beta[j - 1];
  ietl::mult(matrix_, vec2, vec3);
  a = ietl::real(ietl::dot(vec2, vec3));
  vec3 -= a * vec2;
  vec3 -= b * startvector;
  b = ietl::two_norm(vec3);
  vec3 /= b;
  std::swap(vec2, startvector);
  std::swap(vec3, vec2);
  return std::make_pair(a, b);
}

}  // end of namespace ietl.

#endif  // EXTERNAL_IETL_IETL_LANCZOS_H_
