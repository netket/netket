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

#ifndef NETKET_METROPOLISHAMILTONIAN_HH
#define NETKET_METROPOLISHAMILTONIAN_HH

#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <mpi.h>
#include <limits>

namespace netket{

using namespace std;
using namespace Eigen;

//Metropolis sampling generating transitions using the Hamiltonian
template<class WfType,class H> class MetropolisHamiltonian: public AbstractSampler<WfType>{

  WfType & psi_;

  const Hilbert & hilbert_;

  H & hamiltonian_;

  //number of visible units
  const int nv_;

  netket::default_random_engine rgen_;

  //states of visible units
  VectorXd v_;

  VectorXd accept_;
  VectorXd moves_;

  int mynode_;
  int totalnodes_;

  //Look-up tables
  typename WfType::LookupType lt_;


  vector<vector<int>> tochange_;
  vector<vector<double>> newconfs_;
  vector<std::complex<double>> mel_;

  vector<vector<int>> tochange1_;
  vector<vector<double>> newconfs1_;
  vector<std::complex<double>> mel1_;

  VectorXd v1_;

public:

  MetropolisHamiltonian(WfType & psi, H & hamiltonian):
       psi_(psi),hilbert_(psi.GetHilbert()),hamiltonian_(hamiltonian),
       nv_(hilbert_.Size()){
    Init();
  }

  void Init(){
    v_.resize(nv_);

    MPI_Comm_size(MPI_COMM_WORLD, &totalnodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);

    if(!hilbert_.IsDiscrete()){
      if(mynode_==0){
        cerr<<"# Hamiltonian Metropolis sampler works only for discrete Hilbert spaces"<<endl;
      }
      std::abort();
    }

    accept_.resize(1);
    moves_.resize(1);

    Seed();

    Reset(true);


    if(mynode_==0){
      cout<<"# Hamiltonian Metropolis sampler is ready "<<endl;
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
      hilbert_.RandomVals(v_,rgen_);
    }

    psi_.InitLookup(v_,lt_);

    accept_=VectorXd::Zero(1);
    moves_=VectorXd::Zero(1);
  }

  void Sweep(){

    for(int i=0;i<nv_;i++){
      hamiltonian_.FindConn(v_, mel_,tochange_,newconfs_);

      const double w1=tochange_.size();

      std::uniform_int_distribution<int> distrs(0,tochange_.size()-1);
      std::uniform_real_distribution<double> distu;

      //picking a random state to transit to
      int si=distrs(rgen_);


      //Inverse transition
      v1_=v_;
      hilbert_.UpdateConf(v1_,tochange_[si],newconfs_[si]);

      hamiltonian_.FindConn(v1_, mel1_,tochange1_,newconfs1_);

      double w2=tochange1_.size();

      const auto lvd=psi_.LogValDiff(v_,tochange_[si],newconfs_[si],lt_);
      double ratio=std::norm(std::exp(lvd)*w1/w2);

      #ifndef NDEBUG
      const auto psival1=psi_.LogVal(v_);
      if(std::abs(std::exp(psi_.LogVal(v_)-psi_.LogVal(v_,lt_))-1.)>1.0e-8){
        std::cerr<<psi_.LogVal(v_)<<"  and LogVal with Lt is "<<psi_.LogVal(v_,lt_)<<std::endl;
        std::abort();
      }
      #endif

      //Metropolis acceptance test
      if(ratio>distu(rgen_)){
        accept_[0]+=1;
        psi_.UpdateLookup(v_,tochange_[si],newconfs_[si],lt_);
        v_=v1_;

        #ifndef NDEBUG
        const auto psival2=psi_.LogVal(v_);
        if(std::abs(std::exp(psival2-psival1-lvd)-1.)>1.0e-8){
          std::cerr<<psival2-psival1<<" and logvaldiff is "<<lvd<<std::endl;
          std::cerr<<psival2<<" and LogVal with Lt is "<<psi_.LogVal(v_,lt_)<<std::endl;
          std::abort();
        }
        #endif
      }
      moves_[0]+=1;
    }
  }


  VectorXd Visible(){
    return v_;
  }

  void SetVisible(const VectorXd & v){
    v_=v;
  }


  WfType & Psi(){
    return psi_;
  }

  Hilbert & HilbSpace()const{
    return hilbert_;
  }

  VectorXd Acceptance()const{
    VectorXd acc=accept_;
    for(int i=0;i<1;i++){
      acc(i)/=moves_(i);
    }
    return acc;
  }

};


}

#endif
