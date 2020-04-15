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

auto AbstractOperator::GetConnFlattened(Eigen::Ref<const RowMatrix<double>> v,
                                        Eigen::Ref<Eigen::VectorXi> sections)
    -> std::tuple<RowMatrix<double>, Eigen::VectorXcd> {
  Index estimated_size = v.rows() * (2 * GetHilbert().Size());
  RowMatrix<double> vprimes(estimated_size, v.cols());

  Eigen::VectorXcd mels(estimated_size);

  std::vector<Complex> mel;
  std::vector<std::vector<int>> tochange;
  std::vector<std::vector<double>> newconfs;

  Index tot_conn = 0;

  for (auto i = Index{0}; i < v.rows(); ++i) {
    auto vi = Eigen::Ref<const Eigen::VectorXd>{v.row(i)};

    FindConn(vi, mel, tochange, newconfs);

    mels.conservativeResize(tot_conn + mel.size());
    vprimes.conservativeResize(tot_conn + mel.size(), Eigen::NoChange);

    for (std::size_t k = 0; k < tochange.size(); k++) {
      mels(tot_conn + k) = mel[k];

      vprimes.row(tot_conn + k) = vi;
      for (std::size_t c = 0; c < tochange[k].size(); c++) {
        vprimes(tot_conn + k, tochange[k][c]) = newconfs[k][c];
      }
    }

    tot_conn += mel.size();
    sections(i) = tot_conn;
  }

  return std::tuple<RowMatrix<double>, Eigen::VectorXcd>{std::move(vprimes),
                                                         std::move(mels)};
}

void AbstractOperator::GetNConn(Eigen::Ref<const RowMatrix<double>> v,
                                Eigen::Ref<Eigen::VectorXi> n_conn) {
  std::vector<Complex> mel;
  std::vector<std::vector<int>> tochange;
  std::vector<std::vector<double>> newconfs;

  for (auto i = Index{0}; i < v.rows(); ++i) {
    auto vi = Eigen::Ref<const Eigen::VectorXd>{v.row(i)};

    FindConn(vi, mel, tochange, newconfs);
    n_conn(i) = tochange.size();
  }
}

}  // namespace netket
