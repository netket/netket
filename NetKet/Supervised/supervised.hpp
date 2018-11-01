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

#ifndef NETKET_SUPERVISED_CC
#define NETKET_SUPERVISED_CC

#include <memory>
#include <string>

#include "Optimizer/optimizer.hpp"
#include "vmc.hpp"

namespace netket {

class Supervised {
  using VectorType = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

  int batchsize_ = 100;

  Eigen::MatrixXd inputs_;
  Eigen::VectorXcd targets_;

 public:
  explicit Supervised(const json &supervised_pars) {
    // Relevant parameters for supervised learning
    // is stored in supervised_pars.
    CheckFieldExists(supervised_pars, "Supervised");
    const std::string loss_name =
        FieldVal(supervised_pars["Supervised"], "Loss", "Supervised");

    // Input data is encoded in Json file with name "InputFilename".
    auto data_json =
        ReadJsonFromFile(supervised_pars["Supervised"]["InputFilename"]);
    using DataType = Data<double>;
    DataType data(data_json, supervised_pars);

    // Make a machine using the Hilbert space extracted from
    // the data.
    using MachineType = Machine<std::complex<double>>;
    MachineType machine(data.GetHilbert(), supervised_pars);

    if (loss_name == "Overlap") {
      // To do:
      // Check whether we need more advance sampler, i.e. exchange or hop,
      // which are constructed with Graph object provided?
      Sampler<MachineType> sampler(machine, supervised_pars);
      Optimizer optimizer(supervised_pars);

      // To do:
      // Consider adding function (Grad, Init, Run_Supervised) in VMC class,
      // So we do not need to copy all the function VMC class again.
      std::cout << " sampler created, optimizer created \n";
      SupervisedVariationalMonteCarlo vmc(data, sampler, optimizer,
                                          supervised_pars);
      vmc.Run_Supervised();
    } else if (loss_name == "MSE") {
      inputs_.resize(batchsize_, machine.Nvisible());
      targets_.resize(batchsize_);

      VectorType gradC(machine.Npar());

      std::cout << "Running epochs" << std::endl;
      for (int epoch = 0; epoch < 1000; ++epoch) {
        int number_of_batches = ceil(data.Ndata() / float(batchsize_));

        for (int iteration = 0; iteration < number_of_batches; ++iteration) {
          // Generate a batch from the data
          data.GenerateBatch(batchsize_, inputs_, targets_);

          // Compute the gradients
          gradC.setZero();
          std::complex<double> sum_aibi = 0;
          std::complex<double> sum_aiai = 0;
          std::complex<double> sum_bibi = 0;

          for (int x = 0; x < inputs_.rows(); ++x) {
            Eigen::VectorXd config(inputs_.row(x));

            std::complex<double> value = machine.LogVal(config);
            auto partial_gradient = machine.DerLog(config);
            gradC = gradC + partial_gradient * (value - targets_(x));

            // std::cout << "Current gradient: " << gradC << std::endl;
            sum_aibi += std::conj(value)*targets_(x);
            sum_aiai += std::conj(value)*value;
            sum_bibi += std::conj(targets_(x))*targets_(x);
          }

          // Update the parameters
          double alpha = 1e-4;
          std::cout<<" inner "<<sum_aibi/std::sqrt(sum_aiai)/std::sqrt(sum_bibi)<<" grad norm "<<gradC.norm()<<std::endl;
          machine.SetParameters(machine.GetParameters() - alpha * gradC);
        }
      }

    } else {
      std::stringstream s;
      s << "Unknown Supervised loss: " << loss_name;
      throw InvalidInputError(s.str());
    }
  }
};

}  // namespace netket

#endif
