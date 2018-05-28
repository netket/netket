// Copyright 2018 Damian Hofmann - All Rights Reserved.
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

#ifndef NETKET_DENSE_HAMILTONIAN_OPERATOR_HPP
#define NETKET_DENSE_HAMILTONIAN_OPERATOR_HPP

#include <Eigen/Dense>

#include "abstract_matrix_wrapper.hpp"
#include "Hilbert/hilbert_index.hpp"

namespace netket
{

/**
 * This class stores the matrix elements of a given Operator (AbstractHamiltonian
 * or AbstractObservable) as an Eigen dense matrix.
 */
template<class Operator, class WfType = Eigen::VectorXcd>
class DenseMatrixWrapper : public AbstractMatrixWrapper<Operator, WfType>
{
    using Matrix = Eigen::MatrixXcd;

    Matrix matrix_;
    int dim_;

public:
    explicit DenseMatrixWrapper(const Operator &the_operator)
    {
        InitializeMatrix(the_operator);
    }

    WfType Apply(const WfType &state) const override
    {
        return matrix_ * state;
    }

    int GetDimension() const override
    {
        return dim_;
    }

    /**
     * @return An Eigen::MatrixXcd containing the matrix elements of the wrapped operator.
     */
    const Matrix& GetMatrix() const
    {
        return matrix_;
    }

    /**
     * Computes the eigendecomposition of the given matrix.
     * @param options The options are passed directly to the constructor of SelfAdjointEigenSolver.
     * @return An instance of Eigen::SelfAdjointEigenSolver initialized with the wrapped operator and options.
     */
    Eigen::SelfAdjointEigenSolver<Matrix> ComputeEigendecomposition(int options = Eigen::ComputeEigenvectors) const
    {
        return Eigen::SelfAdjointEigenSolver<Matrix>(matrix_, options);
    }

private:
    void InitializeMatrix(const Operator &the_operator)
    {
        const auto& hilbert = the_operator.GetHilbert();
        const HilbertIndex hilbert_index(hilbert);
        dim_ = hilbert_index.NStates();

        matrix_.resize(dim_, dim_);
        matrix_.setZero();

        for(int i = 0; i < dim_; ++i)
        {
            auto v = hilbert_index.NumberToState(i);

            std::vector<std::complex<double>> matrix_elements;
            std::vector<std::vector<int>> connectors;
            std::vector<std::vector<double>> newconfs;
            the_operator.FindConn(v, matrix_elements, connectors, newconfs);

            for(size_t k = 0; k < connectors.size(); ++k)
            {
                auto vk = v;
                hilbert.UpdateConf(vk, connectors[k], newconfs[k]);
                auto j = hilbert_index.StateToNumber(vk);
                matrix_(i, j) += matrix_elements[k];
            }
        }
    }
};

}

#endif //NETKET_DENSE_HAMILTONIAN_OPERATOR_HPP
