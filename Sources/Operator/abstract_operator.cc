// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#include "Operator/abstract_operator.hpp"

namespace netket {

std::tuple<AbstractOperator::MelType, AbstractOperator::ConnectorsType,
           AbstractOperator::NewconfsType>
AbstractOperator::GetConn(VectorConstRefType v) const {
  MelType mel;
  ConnectorsType connectors;
  NewconfsType newconfs;
  FindConn(v, mel, connectors, newconfs);
  return std::make_tuple(std::move(mel), std::move(connectors),
                         std::move(newconfs));
}

void AbstractOperator::ForEachConn(VectorConstRefType v,
                                   ConnCallback callback) const {
  std::vector<Complex> weights;
  std::vector<std::vector<int>> connectors;
  std::vector<std::vector<double>> newconfs;

  FindConn(v, weights, connectors, newconfs);

  for (size_t k = 0; k < connectors.size(); k++) {
    const ConnectorRef conn{weights[k], connectors[k], newconfs[k]};
    callback(conn);
  }
}

}  // namespace netket
