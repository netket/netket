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

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "Utils/all_utils.hpp"
#include "abstract_data.hpp"

#ifndef NETKET_UNIFORM_DATA
#define NETKET_UNIFORM_DATA

namespace netket {

    /** Unifrom data with complex input
     *  Template indended for T=float or double.
     */
    template <typename T>
        class UniformData : public AbstractData<T> {
            using VectorType = typename AbstractData<T>::VectorType;
            using MatrixType = typename AbstractData<T>::MatrixType;

            // number of samples
            int ndata_;

            // number of system size, i.e. visible units
            int nv_;

            // amplitudes with dimension (ndata_, 2)
            // since amp is in general complex
            MatrixType amplitudes;

            // basis with dimension (ndata_, nv_)
            MatrixType basis;


            public:
            using StateType = typename AbstractData<T>::StateType;

            // constructor
            explicit UniformData(const json &pars){
                read_data_from_json(pars);
            }


            int Ndata() const override { return ndata_; }


            /*
            const Hilbert &hilbert_;
            const Hilbert &GetHilbert() const { return hilbert_; }
            int Nvisible() const override { return nv_; }
            void to_json(json &j) const override {
                j["Data"]["Name"] = "UniformData";
                j["Data"]["Nvisible"] = nv_;
            }
            */

            //
            // methods for unifrom data
            //
            void read_data_from_json(const json &pars) {
                // should add throw exception here //
                amplitudes = pars["samples"]["amp"];
                ndata_ = amplitudes.rows();
                nv_ = amplitudes.cols();
                basis = pars["samples"]["basis"];
                //if (nv_ != pars["system_size"]){
                //    // Not sure whether this is neccessary.
                //    throw InvalidInputError("Data and system_size mismatch !");
                //}
                std::cout<< " read in amplitudes as array \n" << amplitudes << "\n"
                    << "read in basis as array \n" << basis << "\n";
            }


        };

}  // namespace netket

#endif
