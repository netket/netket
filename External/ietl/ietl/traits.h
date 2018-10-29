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

/* $Id: traits.h,v 1.2 2004/02/15 23:30:42 troyer Exp $ */
/* Modified by Alexander Wietek, 2018/10/22 */

#ifndef EXTERNAL_IETL_IETL_TRAITS_H_
#define EXTERNAL_IETL_IETL_TRAITS_H_

#include <complex>

namespace ietl {
  template <class T>
  struct number_traits {
    typedef T magnitude_type;
  };

  template <class T>
  struct number_traits<std::complex<T> > {
    typedef T magnitude_type;
  };

  template <class VS>
  struct vectorspace_traits {
    typedef typename VS::vector_type vector_type;
    typedef typename VS::size_type size_type;
    typedef typename VS::scalar_type scalar_type;
    typedef typename number_traits<scalar_type>::magnitude_type magnitude_type;
  };
}

#endif  // EXTERNAL_IETL_IETL_TRAITS_H_
