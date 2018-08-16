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

namespace netket {

    class Supervised {
        public:
            explicit Supervised(const json &pars) {
                CheckFieldExists(pars, "Supervised");
                const std::string loss_name = FieldVal(pars["Supervised"], "Loss", "Supervised");

                if (loss_name == "L2" ||
                        loss_name == "cosine") {

                    // Graph graph(pars);

                    // Hamiltonian hamiltonian(graph, pars);

                    // using MachineType = Machine<std::complex<double>>;
                    // MachineType machine(graph, hamiltonian, pars);
                    using DataType = UniformData<double>;
                    auto pars_data = ReadJsonFromFile(pars["Supervised"]["InputFilename"]);
                    DataType data(pars_data);

                        // Sampler<MachineType> sampler(graph, hamiltonian, machine, pars);

                        // Stepper stepper(pars);

                        // GroundState le(hamiltonian, sampler, stepper, pars);
                }
                else {
                    std::stringstream s;
                    s << "Unknown Supervised loss: " << loss_name;
                    throw InvalidInputError(s.str());
                }
            }

    };

}  // namespace netket

#endif
