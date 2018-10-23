/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2002 by Prakash Dayal <prakash@comp-phys.org>,
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

/* $Id: complex.h,v 1.5 2003/09/13 10:30:24 troyer Exp $ */
/* Modified by Alexander Wietek, 2018/10/22 */

#ifndef EXTERNAL_IETL_IETL_COMPLEX_H_
#define EXTERNAL_IETL_IETL_COMPLEX_H_

#include <complex>

namespace ietl {

  template <class T>
  struct real_type {
    typedef T type;
  };

  template <class T>
  struct real_type<std::complex<T> > {
    typedef T type;
  };

  template <class T> T real (T x) { return x; }
  template <class T> T real (std::complex<T> x) { return x.real(); }
  template <class T> T conj (T x) { return x; }
  template <class T> std::complex<T> conj(std::complex<T> x)
  { return std::conj(x); }
}
#endif  // EXTERNAL_IETL_IETL_COMPLEX_H_
