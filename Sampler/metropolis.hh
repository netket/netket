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

#ifndef NETKET_METROPOLISSPIN_HH
#define NETKET_METROPOLISSPIN_HH

#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <mpi.h>

namespace netket{

using namespace std;
using namespace Eigen;

//Metropolis sampling for binary Rbms
//flippng k random visible/hidden spins
template<class RbmType> class Metropolis{

  //number of visible units
  const int nv_;

  //number of hidden units
  const int nh_;

  std::mt19937 rgen_;

  RbmType & rbm_;

  //states of visible and hidden units
  VectorXd v_;
  VectorXd h_;


  VectorXd accept_;
  VectorXd moves_;

  int mynode_;
  int totalnodes_;

  std::vector<int> visiblen_;
  std::vector<int> hiddenn_;

  int nflips_;

  //additional (optional) gibbs sweep
  const bool dogibbs_;

  //probabilities for gibbs sampling
  VectorXd probv_;
  VectorXd probh_;

public:

  Metropolis(RbmType & rbm,int nflips,bool dogibbs=false):
       rbm_(rbm),nv_(rbm.Nvisible()),nh_(rbm.Nhidden()),nflips_(nflips),dogibbs_(dogibbs){

    v_.resize(nv_);
    h_.resize(nh_);

    if(dogibbs_){
      probv_.resize(nv_);
      probh_.resize(nh_);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    visiblen_.resize(nv_);
    hiddenn_.resize(nh_);

    for(int i=0;i<nv_;i++){
      visiblen_[i]=i;
    }
    for(int j=0;j<nh_;j++){
      hiddenn_[j]=j;
    }

    accept_.resize(2);
    moves_.resize(2);

    Seed();

    rbm_.RandomVals(v_,rgen_);
    rbm_.RandomVals(h_,rgen_);

    if(mynode_==0){
      cout<<"# Metropolis sampler is ready "<<endl;
    }

  }

  void Seed(int baseseed=0){
    std::random_device rd;
    vector<int> seeds(totalnodes_);

    if(mynode_==0){
      for(int i=0;i<totalnodes_;i++){
        seeds[i]=rd()+baseseed;
      }
    }

    SendToAll(seeds);

    rgen_.seed(seeds[mynode_]);
  }


  void Reset(bool initrandom=false){
    if(initrandom){
      rbm_.RandomVals(v_,rgen_);
      rbm_.RandomVals(h_,rgen_);
    }

    accept_=VectorXd::Zero(2);
    moves_=VectorXd::Zero(2);
  }

  void Sweep(){

    std::shuffle(visiblen_.begin(), visiblen_.end(), rgen_);
    std::shuffle(hiddenn_.begin(), hiddenn_.end(), rgen_);

    int i=0;
    int j=0;

    assert(nflips_<nv_ && nflips_<nh_);

    vector<int> toflip(nflips_);
    std::uniform_real_distribution<double> distu;

    while(i<nv_ && j<nv_){

      if(i<nv_){
        for(int f=0;f<nflips_;f++){
          toflip[f]=visiblen_[i];
          i++;
        }

        double ratio=rbm_.RatioVisibleFlip(v_,h_,toflip);

        if(ratio>distu(rgen_)){
          accept_[0]+=1;
          FlipVisible(toflip);
        }

        moves_[0]+=1;
      }
      if(j<nh_){
        for(int f=0;f<nflips_;f++){
          toflip[f]=hiddenn_[j];
          j++;
        }
        double ratio=rbm_.RatioHiddenFlip(v_,h_,toflip);

        if(ratio>distu(rgen_)){
          accept_[1]+=1;
          FlipHidden(toflip);
        }

        moves_[1]+=1;
      }

    }

    if(dogibbs_){
      GibbsSweep();
    }

  }

  void GibbsSweep(){
    rbm_.ProbHiddenGivenVisible(v_,probh_);
    rbm_.RandomValsWithProb(h_,probh_,rgen_);

    rbm_.ProbVisibleGivenHidden(h_,probv_);
    rbm_.RandomValsWithProb(v_,probv_,rgen_);
  }


  VectorXd Visible(){
    return v_;
  }


  VectorXd Hidden(){
    return h_;
  }

  void SetVisible(const VectorXd & v){
    v_=v;
  }

  void SetHidden(const VectorXd & h){
    h_=h;
  }

  void FlipVisible(vector<int> & toflip){
    for(int i=0;i<toflip.size();i++){
      v_(toflip[i])*=-1;
    }
  }

  void FlipHidden(vector<int> & toflip){
    for(int i=0;i<toflip.size();i++){
      h_(toflip[i])*=-1;
    }
  }

  RbmType & Rbm(){
    return rbm_;
  }

  RbmType & Psi(){
    return rbm_;
  }

  VectorXd Acceptance()const{
    VectorXd acc=accept_;
    for(int i=0;i<2;i++){
      acc(i)/=moves_(i);
    }
    return acc;
  }

};


}

#endif
