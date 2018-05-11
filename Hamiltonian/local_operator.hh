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

#ifndef NETKET_LOCAL_OPERATOR_HH
#define NETKET_LOCAL_OPERATOR_HH

#include <vector>
#include <iostream>
#include <cassert>
#include <limits>
#include <algorithm>
#include <map>
#include <Eigen/Dense>
#include <complex>
#include <mpi.h>

namespace netket{

/**
    Class for local operators acting on a list of sites and for generic local
    Hilbert spaces.
*/

using namespace std;
using namespace Eigen;

class LocalOperator{

public:
  using MatType=std::vector<std::vector<std::complex<double>>>;

private:
  const Hilbert & hilbert_;
  MatType mat_;

  std::vector<int> sites_;

  std::map<vector<double>,int> invstate_;
  int localsize_;

  vector<vector<double>> states_;
  vector<vector<int>> connected_;

public:

  LocalOperator(const Hilbert & hilbert,const MatType & mat,
           const std::vector<int> & sites):
    hilbert_(hilbert),mat_(mat),sites_(sites){

    Init();
  }

  void Init(){

    if(!hilbert_.IsDiscrete()){
      cerr<<"Cannot construct operators on infinite local hilbert spaces"<<endl;
      std::abort();
    }

    if(*std::max_element(sites_.begin(),sites_.end())>=hilbert_.Size()
      || *std::min_element(sites_.begin(),sites_.end())<0){
      cerr<<"Operator acts on an invalid set of sites"<<endl;
      std::abort();
    }

    auto localstates=hilbert_.LocalStates();
    localsize_=localstates.size();

    //Finding the non-zero matrix elements
    const double epsilon=1.0e-6;

    connected_.resize(mat_.size());

    if(mat_.size()!=std::pow(localsize_,sites_.size())){
      cerr<<"Matrix size in operator is inconsistent with Hilbert space"<<endl;
      std::abort();
    }

    for(std::size_t i=0;i<mat_.size();i++){
      for(std::size_t j=0;j<mat_[i].size();j++){

        if(mat_.size()!=mat_[i].size()){
          cerr<<"Matrix size in operator is inconsistent with Hilbert space"<<endl;
          std::abort();
        }

        if(i!=j && std::abs(mat_[i][j])>epsilon){
          connected_[i].push_back(j);
        }
      }
    }

    //Construct the mapping
    //Internal index -> State
    vector<double> st(sites_.size(),0);

    do{
      states_.push_back(st);
    }
    while( netket::next_variation(st.begin(), st.end(),localsize_-1) );

    for(std::size_t i=0;i<states_.size();i++){
      for(std::size_t k=0;k<states_[i].size();k++){
        states_[i][k]=localstates[states_[i][k]];
      }
    }

    //Now construct the inverse mapping
    //State -> Internal index
    int k=0;
    for(auto state : states_){
      invstate_[state]=k;
      k++;
    }

    assert(k==mat_.size());

  }

  void FindConn(const VectorXd & v,
    vector<std::complex<double>> & mel,
    vector<vector<int>> & connectors,
    vector<vector<double>> & newconfs)const{

    assert(v.size()==hilbert_.Size());

    connectors.clear();
    newconfs.clear();
    mel.resize(0);

    AddConn(v,mel,connectors,newconfs);

  }

  void AddConn(const VectorXd & v,
    vector<std::complex<double>> & mel,
    vector<vector<int>> & connectors,
    vector<vector<double>> & newconfs)const{

    if(mel.size()==0){
      connectors.resize(1);
      newconfs.resize(1);
      mel.resize(1);

      mel[0]=0;
      connectors[0].resize(0);
      newconfs[0].resize(0);
    }

    int st1=StateNumber(v);
    assert(st1<mat_.size());
    assert(st1<connected_.size());

    mel[0]+=(mat_[st1][st1]);

    //off-diagonal part
    for(auto st2 : connected_[st1]){
      connectors.push_back(sites_);
      assert(st2<states_.size());
      newconfs.push_back(states_[st2]);
      mel.push_back(mat_[st1][st2]);
    }

  }

  inline int StateNumber(const VectorXd & v)const{
    vector<double> state(sites_.size());
    for(std::size_t i=0;i<sites_.size();i++){
      state[i]=v(sites_[i]);
    }
    return invstate_.at(state);
  }

};

}
#endif
