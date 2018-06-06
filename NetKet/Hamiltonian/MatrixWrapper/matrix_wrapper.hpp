#ifndef MATRIX_WRAPPER_HPP
#define MATRIX_WRAPPER_HPP

#include <memory>

#include "Graph/graph.hpp"
#include "Hamiltonian/hamiltonian.hpp"
#include "Utils/json_helper.hpp"

#include "dense_matrix_wrapper.hpp"
#include "direct_matrix_wrapper.hpp"
#include "sparse_matrix_wrapper.hpp"

namespace netket {

template<class Wrapped>
std::unique_ptr<AbstractMatrixWrapper<Wrapped>>
ConstructMatrixWrapper(const json &pars, const Wrapped& wrapped)
{
    using WrapperPtr = std::unique_ptr<AbstractMatrixWrapper<Wrapped>>;

    std::string wrapper_name = FieldOrDefaultVal<json, std::string>(pars, "MatrixWrapper", "Sparse");
    if(wrapper_name == "Sparse")
    {
        return WrapperPtr(new SparseMatrixWrapper<Wrapped>(wrapped));
    }
    else if(wrapper_name == "Dense")
    {
        return WrapperPtr(new DenseMatrixWrapper<Wrapped>(wrapped));
    }
    else
    {
        std::cout << "Unknown MatrixWrapper: " << wrapper_name << std::endl;
        std::abort();
    }
}

}

#endif // MATRIX_WRAPPER_HPP
