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

#ifndef NETKET_HAMILTONIAN_HH
#define NETKET_HAMILTONIAN_HH

#include <memory>

#include "abstract_hamiltonian.hh"
#include "ising.hh"
#include "heisenberg.hh"
#include "bosonhubbard.hh"
#include "custom_hamiltonian.hh"

namespace netket{

class Hamiltonian:public AbstractHamiltonian{
  using Ptype=std::unique_ptr<AbstractHamiltonian>;

  Ptype h_;

public:

  Hamiltonian(const Graph & graph,const json & pars){

    if(!FieldExists(pars,"Hamiltonian")){
      cerr<<"Hamiltonian is not defined in the input"<<endl;
      std::abort();
    }

    if(FieldExists(pars["Hamiltonian"],"Name")){
      if(pars["Hamiltonian"]["Name"]=="Ising"){
        h_=Ptype(new Ising<Graph>(graph,pars));
      }
      else if(pars["Hamiltonian"]["Name"]=="Heisenberg"){
        h_=Ptype(new Heisenberg<Graph>(graph,pars));
      }
      else if(pars["Hamiltonian"]["Name"]=="BoseHubbard"){
        h_=Ptype(new BoseHubbard<Graph>(graph,pars));
      }
      else{
        cout<<"Hamiltonian name not found"<<endl;
        std::abort();
      }
    }
    else{
      h_=Ptype(new CustomHamiltonian(pars));
    }
  }

  void FindConn(const VectorXd & v,
    vector<std::complex<double>> & mel,
    vector<vector<int>> & connectors,
    vector<vector<double>> & newconfs)
  {
      return h_->FindConn(v,mel,connectors,newconfs);
  }

  const Hilbert & GetHilbert()const{
    return h_->GetHilbert();
  }
};
}
#endif
