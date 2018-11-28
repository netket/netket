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

#ifndef NETKET_UNSUPERVISED_CC
#define NETKET_UNSUPERVISED_CC

#include <memory>

#include "Hamiltonian/MatrixWrapper/matrix_wrapper.hpp"
#include "Observable/observable.hpp"
#include "Optimizer/optimizer.hpp"

#include "unsupervised_learning.hpp"
#include "test.hpp"
#include "quantum_state_reconstruction.hpp"
namespace netket {

class Unsupervised {
 public:
  explicit Unsupervised(const json &pars) {
    std::string method_name;

    if (FieldExists(pars, "Unsupervised")) {
      method_name = FieldVal(pars["Unsupervised"], "Method", "Unsupervised");
    } else if (FieldExists(pars, "Learning")) {
      method_name = FieldVal(pars["Learning"], "Method", "Learning");
      // DEPRECATED (to remove for v2.0.0)
      WarningMessage()
          << "Use of the Learning section is "
             "deprecated.\n Please use the dedicated Unsupervised section.\n";
    } else {
      std::stringstream s;
      s << "The Unsupervised section has not been specified.\n";
      throw InvalidInputError(s.str());
    }

    Graph graph(pars);
    //Hamiltonian hamiltonian(graph, pars);

    if (method_name == "Gd" || method_name == "Sr") {
      using MachineType = Machine<std::complex<double>>;
      Hilbert hilbert(pars);
      MachineType machine(graph, hilbert, pars);

      Sampler<MachineType> sampler(graph, machine, pars);
      Optimizer optimizer(pars);

      //UnsupervisedLearning unsupervised(sampler,optimizer,pars);
      //unsupervised.TestDerKL();
      //unsupervised.Run();
      QuantumStateReconstruction QSR(sampler,optimizer,pars);
      QSR.TestDerKL();
      QSR.TestDerNLL();
      //QSR.Run();
      //Test t(sampler,pars);
      //t.TestDerKL();

    } else {
      std::stringstream s;
      s << "Unknown Unsupervised method: " << method_name;
      throw InvalidInputError(s.str());
    }
  }

};

}  // namespace netket

#endif
