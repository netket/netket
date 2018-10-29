/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2002 by Rene Villiger <rvilliger@smile.ch>,
*                            Prakash Dayal <prakash@comp-phys.org>,
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

/* $Id: fmatrix.h,v 1.8 2003/04/24 13:49:40 renev Exp $ */
/* Modified by Alexander Wietek, 2018/10/22 */

#ifndef EXTERNAL_IETL_IETL_FMATRIX_H_
#define EXTERNAL_IETL_IETL_FMATRIX_H_

#include <cstddef>

#undef minor
#undef data

namespace ietl {

  template <class T>
  class FortranMatrix {
  private:
    T* p;
    std::size_t n_;
    std::size_t m_;
  public:
    typedef std::size_t size_type;
    FortranMatrix(size_type n, size_type m) : n_(n), m_(m) { p = new T[m*n]; }
    ~FortranMatrix() { delete[] p; }
    T* data() { return p; }
    const T* data() const { return p; }
    T operator()(size_type i, size_type j) const { return p[i+j*n_]; }
    T& operator()(size_type i, size_type j) { return p[i+j*n_]; }
    void resize(size_type n, size_type m) {
      m_ = m;
      n_ = n;
      delete[] p;
      p = new T[m*n];
    }
    size_type nrows() { return n_; }
    size_type ncols() { return m_; }
    size_type minor() { return n_; }
  };
}
#endif  // EXTERNAL_IETL_IETL_FMATRIX_H_
