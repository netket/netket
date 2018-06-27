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

#ifndef NETKET_ABSTRACTDATA_HPP
#define NETKET_ABSTRACTDATA_HPP

#include <Eigen/Dense>
#include <complex>
#include <fstream>
#include "Utils/json_utils.hpp"
#include "Utils/random_utils.hpp"
#include <vector>

namespace netket {
    /**
      Abstract class for Data.
      */
    template <typename T>
        class AbstractData {
            public:
                using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
                using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
                using StateType = T;

                /**
                  Member function returning the number of samples in the data.
                  */
                virtual int Ndata() const = 0;

                /*
                virtual void to_json(json &j) const = 0;
                virtual void from_json(const json &j) = 0;

                void Save(std::string filename) const {
                    std::ofstream filewf(filename);

                    json j;
                    to_json(j);
                    filewf << j << std::endl;

                    filewf.close();
                }
                */

                virtual ~AbstractData() {}
        };
}  // namespace netket

#endif
