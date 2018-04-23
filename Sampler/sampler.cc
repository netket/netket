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

#ifndef NETKET_SAMPLER_CC
#define NETKET_SAMPLER_CC

namespace netket{

template<class WfType> class Sampler:public AbstractSampler<WfType>{
  AbstractSampler<WfType> * s_;
public:
  Sampler(Graph & graph,Hamiltonian<Graph> & hamiltonian,WfType & psi,const json & pars){

    if(!FieldExists(pars,"Sampler")){
      cerr<<"Sampler is not defined in the input"<<endl;
      std::abort();
    }

    if(!FieldExists(pars["Sampler"],"Name")){
      cerr<<"Sampler Name is not defined in the input"<<endl;
      std::abort();
    }

    if(pars["Sampler"]["Name"]=="MetropolisLocal"){
      s_=new MetropolisLocal<WfType>(graph,psi,pars);
    }
    else if(pars["Sampler"]["Name"]=="MetropolisLocalPt"){
      s_=new MetropolisLocalPt<WfType>(graph,psi,pars);
    }
    else if(pars["Sampler"]["Name"]=="MetropolisExchange"){
      s_=new MetropolisExchange<WfType>(graph,psi,pars);
    }
    else if(pars["Sampler"]["Name"]=="MetropolisExchangePt"){
      s_=new MetropolisExchangePt<WfType>(graph,psi,pars);
    }
    else if(pars["Sampler"]["Name"]=="MetropolisHop"){
      s_=new MetropolisHop<WfType>(graph,psi,pars);
    }
    else if(pars["Sampler"]["Name"]=="MetropolisHamiltonian"){
      s_=new MetropolisHamiltonian<WfType,Hamiltonian<Graph>>(graph,psi,hamiltonian,pars);
    }
    else if(pars["Sampler"]["Name"]=="MetropolisHamiltonianPt"){
      s_=new MetropolisHamiltonianPt<WfType,Hamiltonian<Graph>>(graph,psi,hamiltonian,pars);
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
