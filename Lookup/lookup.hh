// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_LOOKUP_HH
#define NETKET_LOOKUP_HH

#include <Eigen/Dense>
#include <vector>
#include <cassert>


namespace netket{

  //Generic look-up table
  template<class T> class Lookup{

    using VectorType=Matrix<T,Dynamic,1>;
    using MatrixType=Matrix<T,Dynamic,Dynamic>;

    vector<VectorType> v_;
    vector<MatrixType> m_;

  public:

    int AddVector(int a){
      v_.push_back(VectorType(a));
      return v_.size()-1;
    }

    int AddMatrix(int a,int b){
      m_.push_back(MatrixType(a,b));
      return m_.size()-1;
    }

    int VectorSize(){
      return v_.size();
    }

    int MatrixSize(){
      return m_.size();
    }

    VectorType & V(int i){
      assert(i<v_.size() && i>=0);
      return v_[i];
    }

    const VectorType & V(int i)const{
      assert(i<v_.size() && i>=0);
      return v_[i];
    }

    MatrixType & M(int i){
      assert(i<m_.size() && i>=0);
      return m_[i];
    }

    const MatrixType & M(int i)const{
      assert(i<m_.size() && i>=0);
      return m_[i];
    }
  };
}


#endif
