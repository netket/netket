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

#ifndef NETKET_SAMPLER_HH
#define NETKET_SAMPLER_HH

#include <memory>
#include "abstract_sampler.hh"
#include "metropolis_local.hh"
#include "metropolis_exchange.hh"
#include "metropolis_exchange_pt.hh"
#include "metropolis_local_pt.hh"
#include "metropolis_hop.hh"
#include "metropolis_hamiltonian.hh"
#include "metropolis_hamiltonian_pt.hh"

namespace netket{

template<class WfType> class Sampler:public AbstractSampler<WfType>{

  using Ptype=std::unique_ptr<AbstractSampler<WfType>>;
  Ptype s_;

public:
  Sampler(Graph & graph,Hamiltonian & hamiltonian,WfType & psi,const json & pars){

    if(!FieldExists(pars,"Sampler")){
      cerr<<"Sampler is not defined in the input"<<endl;
      std::abort();
    }

    if(!FieldExists(pars["Sampler"],"Name")){
      cerr<<"Sampler Name is not defined in the input"<<endl;
      std::abort();
    }

    if(pars["Sampler"]["Name"]=="MetropolisLocal"){
      s_=Ptype(new MetropolisLocal<WfType>(graph,psi,pars));
    }
    else if(pars["Sampler"]["Name"]=="MetropolisLocalPt"){
      s_=Ptype(new MetropolisLocalPt<WfType>(graph,psi,pars));
    }
    else if(pars["Sampler"]["Name"]=="MetropolisExchange"){
      s_=Ptype(new MetropolisExchange<WfType>(graph,psi,pars));
    }
    else if(pars["Sampler"]["Name"]=="MetropolisExchangePt"){
      s_=Ptype(new MetropolisExchangePt<WfType>(graph,psi,pars));
    }
    else if(pars["Sampler"]["Name"]=="MetropolisHop"){
      s_=Ptype(new MetropolisHop<WfType>(graph,psi,pars));
    }
    else if(pars["Sampler"]["Name"]=="MetropolisHamiltonian"){
      s_=Ptype(new MetropolisHamiltonian<WfType,Hamiltonian>(graph,psi,hamiltonian,pars));
    }
    else if(pars["Sampler"]["Name"]=="MetropolisHamiltonianPt"){
      s_=Ptype(new MetropolisHamiltonianPt<WfType,Hamiltonian>(graph,psi,hamiltonian,pars));
    }
    else{
      cout<<"Sampler not found"<<endl;
      std::abort();
    }
  }

  void Reset(bool initrandom=false){
    return s_->Reset(initrandom);
  }
  void Sweep(){
    return s_->Sweep();
  }
  VectorXd Visible(){
    return s_->Visible();
  }
  void SetVisible(const VectorXd & v){
    return s_->SetVisible(v);
  }
  WfType & Psi(){
    return s_->Psi();
  }
  VectorXd Acceptance()const{
    return s_->Acceptance();
  }

};
}

#endif
