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

#include "Machine/abstract_machine.hpp"

namespace netket {

void AbstractOperator::FindConn(VectorConstRefType v,
                                Eigen::SparseMatrix<double>& delta_v,
                                Eigen::VectorXcd& mels) const {
  delta_v.setZero();

  std::vector<Complex> weights;
  std::vector<std::vector<int>> connectors;
  std::vector<std::vector<double>> newconfs;

  FindConn(v, weights, connectors, newconfs);

  delta_v.resize(connectors.size(), v.size());

  mels.resize(connectors.size());

  for (size_t k = 0; k < connectors.size(); k++) {
    for (size_t c = 0; c < connectors[k].size(); c++) {
      delta_v.insert(k, connectors[k][c]) =
          newconfs[k][c] - v[connectors[k][c]];
    }
    mels(k) = weights[k];
  }
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

auto AbstractOperator::GetConn(Eigen::Ref<const RowMatrix<double>> v)
    -> std::tuple<std::vector<RowMatrix<double>>,
                  std::vector<Eigen::VectorXcd>> {
  std::vector<RowMatrix<double>> vprimes(v.rows());
  std::vector<Eigen::VectorXcd> mels(v.rows());

  std::vector<Complex> mel;
  std::vector<std::vector<int>> tochange;
  std::vector<std::vector<double>> newconfs;

  for (auto i = Index{0}; i < v.rows(); ++i) {
    auto vi = Eigen::Ref<const Eigen::VectorXd>{v.row(i)};

    FindConn(vi, mel, tochange, newconfs);

    mels[i] = Eigen::Map<const Eigen::VectorXcd>(&mel[0], mel.size());

    vprimes[i] = vi.transpose().colwise().replicate(mel.size());

    for (std::size_t k = 0; k < tochange.size(); k++) {
      for (std::size_t c = 0; c < tochange[k].size(); c++) {
        vprimes[i](k, tochange[k][c]) = newconfs[k][c];
      }
    }
  }

  return std::tuple<std::vector<RowMatrix<double>>,
                    std::vector<Eigen::VectorXcd>>{std::move(vprimes),
                                                   std::move(mels)};
}
}  // namespace netket
