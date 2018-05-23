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

#ifndef NETKET_DIRECT_HAMILTONIAN_OPERATOR_HPP
#define NETKET_DIRECT_HAMILTONIAN_OPERATOR_HPP

#include "abstract_matrix_wrapper.hpp"
#include "Hilbert/hilbert_index.hpp"

namespace netket
{

/**
 * This class wraps a given Operator (AbstractHamiltonian or AbstractObservable).
 * The matrix elements are not stored separately but are computed from Operator::FindConn
 * every time Apply is called.
 */
template<class Operator, class WfType = Eigen::VectorXcd>
class DirectMatrixWrapper : public AbstractMatrixWrapper<WfType>
{
    const Operator& operator_;
    size_t dim_;

public:
    explicit DirectMatrixWrapper(const Operator& the_operator)
        : operator_(the_operator),
          dim_(HilbertIndex(the_operator.GetHilbert()).NStates())
    {
    }

    WfType Apply(const WfType& state) const override
    {
        const auto& hilbert = operator_.GetHilbert();
        const HilbertIndex hilbert_index(hilbert);

        WfType result(dim_);
        result.setZero();

        for(int i = 0; i < dim_; ++i)
        {
            auto v = hilbert_index.NumberToState(i);

            std::vector<std::complex<double>> matrix_elements;
            std::vector<std::vector<int>> connectors;
            std::vector<std::vector<double>> newconfs;
            operator_.FindConn(v, matrix_elements, connectors, newconfs);

            for(size_t k = 0; k < connectors.size(); ++k)
            {
                auto vk = v;
                hilbert.UpdateConf(vk, connectors[k], newconfs[k]);
                auto j = hilbert_index.StateToNumber(vk);

                result(j) += matrix_elements[k] * state(i);
            }
        }
        return result;
    }

    int GetDimension() const override
    {
        return dim_;
    }
};

#endif //NETKET_DIRECT_HAMILTONIAN_OPERATOR_HPP
