#include "Stats/stats.hpp"

#include <iomanip>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Stats/mc_stats.hpp"
#include "Stats/obs_manager.hpp"
#include "Utils/exceptions.hpp"
#include "common_types.hpp"

namespace py = pybind11;

namespace netket {
namespace detail {

py::dict GetItem(const ObsManager& self, const std::string& name) {
  py::dict dict;
  self.InsertAllStats(name, dict);
  return dict;
}

int GetPrecision(double /* value */, double error) {
  const int loge = std::floor(std::log10(std::abs(error)));
  return std::max(1 - loge, 0);
}

int GetPrecision(Complex value, double error) {
  return GetPrecision(std::max(std::abs(value.real()), std::abs(value.imag())),
                      error);
}

void AddStatsModule(py::module m) {
  auto subm = m.def_submodule("stats");

  py::class_<ObsManager>(subm, "ObsManager")
      .def("__getitem__", &detail::GetItem, py::arg("name"))
      .def("__getattr__", &detail::GetItem, py::arg("name"))
      .def("__contains__", &ObsManager::Contains, py::arg("name"))
      .def("__len__", &ObsManager::Size)
      .def("keys", &ObsManager::Names)
      .def("__repr__", [](const ObsManager& self) {
        std::string s("<netket.stats.ObsManager: size=");
        auto size = self.Size();
        s += std::to_string(size);
        if (size > 0) {
          s += " [";
          for (const auto& name : self.Names()) {
            s += name + ", ";
          }
          // remove last comma + space:
          s.pop_back();
          s.pop_back();

          s += "]";
        }
        return s + ">";
      });

  auto as_dict = [](const Stats& self) {
    py::dict d;
    d["Mean"] = self.mean;
    d["Sigma"] = self.error_of_mean;
    d["Variance"] = self.variance;
    d["R"] = self.R;
    d["TauCorr"] = self.correlation;
    return d;
  };

  py::class_<Stats>(subm, "Stats")
      .def_readonly("mean", &Stats::mean)
      .def_readonly("error_of_mean", &Stats::error_of_mean)
      .def_readonly("variance", &Stats::variance)
      .def_readonly("tau_corr", &Stats::correlation)
      .def_readonly("R", &Stats::R)
      .def("__repr__",
           [](const Stats& self) {
             std::ostringstream stream;
             const double imag = self.mean.imag();
             const int precision = GetPrecision(self.mean, self.error_of_mean);
             // clang-format off
             stream << std::fixed << std::setprecision(precision)
                    << "(" << self.mean.real()
                    << (imag >= 0 ? " + " : " - ") << std::abs(imag)
                    << "i) ± " << self.error_of_mean
                    << " [var=" << self.variance
                    << ", R=" << std::setprecision(5) << self.R
                    << "]";
             // clang-format on
             return stream.str();
           })
      .def("_asdict", as_dict)  //< compatibility with namedtuple
      .def("asdict", as_dict);

  subm.def("statistics",
           [](py::array_t<Complex, py::array::c_style> local_values) {
             switch (local_values.ndim()) {
               case 2:
                 return Statistics(
                     Eigen::Map<const Eigen::VectorXcd>{local_values.data(),
                                                        local_values.size()},
                     /*n_chains=*/local_values.shape(1));
               case 1:
                 return Statistics(
                     Eigen::Map<const Eigen::VectorXcd>{local_values.data(),
                                                        local_values.size()},
                     /*n_chains=*/1);
               default:
                 NETKET_CHECK(false, InvalidInputError,
                              "local_values has wrong dimension: "
                                  << local_values.ndim()
                                  << "; expected either 1 or 2.");
             }  // end switch
           },
           py::arg{"values"}.noconvert(),
           R"EOF(Computes some statistics (see `Stats` class) of a sequence of
            local estimators obtained from Monte Carlo sampling.

            Args:
                values: A tensor of local estimators. It can be either a rank-1
                    or a rank-2 tensor of `complex128`. Rank-1 tensors represent
                    data from a single Markov Chain, so e.g. `error_on_mean` will
                    be `None`.

                    Rank-2 tensors should have shape `(N, M)` where `N` is the
                    number of samples in one Markov Chain and `M` is the number
                    of Markov Chains. Data should be in row major order.)EOF");

  subm.def(
      "covariance_sv",
      [](py::array_t<Complex, py::array::c_style> s_values,
         py::array_t<Complex, py::array::c_style> v_values, bool center_s) {
        Eigen::Map<const VectorXcd> s_vector{s_values.data(), s_values.size()};

        // Compute S -> S - 𝔼[S]
        const Complex mean = [&s_vector, center_s]() -> Complex {
          if (center_s) {
            Complex mean = s_vector.mean();
            MeanOnNodes(mean);
            return mean;
          } else {
            return 0.;
          }
        }();

        switch (s_values.ndim()) {
          case 2:
            NETKET_CHECK(v_values.ndim() == 3, InvalidInputError,
                         "v_values has wrong dimension: " << v_values.ndim()
                                                          << "; expected 3.");
            return product_sv(
                s_vector.array() - mean,
                Eigen::Map<const RowMatrix<Complex>>{
                    v_values.data(), v_values.shape(0) * v_values.shape(1),
                    v_values.shape(2)});
          case 1:
            NETKET_CHECK(v_values.ndim() == 2, InvalidInputError,
                         "v_values has wrong dimension: " << v_values.ndim()
                                                          << "; expected 2.");
            return product_sv(
                s_vector.array() - mean,
                Eigen::Map<const RowMatrix<Complex>>{
                    v_values.data(), v_values.shape(0), v_values.shape(1)});
          default:
            NETKET_CHECK(false, InvalidInputError,
                         "s_values has wrong dimension: "
                             << s_values.ndim() << "; expected either 1 or 2.");
        }  // end switch
      },
      py::arg{"s"}.noconvert(), py::arg{"v"}.noconvert(),
      py::arg{"center_s"} = true,
      R"EOF(Computes the covariance of two random variables S, V from MCMC
            data. Note that while S has to be a complex scalar variable,
            V = (v[1], ..., v[m]) can be vector-valued. This is because this
            function is designed to be used for computing the gradient of
            expectation values <O>, where S is the local estimator of O and
            V the vector of log-derivatives of the wavefunction.

            This function estimates R = (r[1], ..., r[m]) where
                r[i] = Cov[v[i], S] = 𝔼[conj(v[i]) * (S - 𝔼[S])].

            Args:
                s_values: A vector (or a matrix) of samples of S with shape
                    `(N, M)` where `N` is the number of samples in every Markov
                    Chain and `M` is the number of Markov Chains.
                    A vector is considered to be an `(N, 1)` matrix.
                v_values: A matrix (or a rank-3 array) of samples of V.
                    Each row of the matrix must correspond to one component v[i].
                    For a rank-3 array, its shape is `(N, M, m)` where `N`
                    is the number of samples, `M` is the number of Markov Chains,
                    and `m` is the dimension of V. A `(N, m)` matrix
                    is treated an an `(N, 1, m)` array.
                center_s (bool=True): Whether S should be centered, i.e.,
                    computing S - 𝔼[S]. If set to False, this function will
                    only return the mathematical covariance if either S or all
                    v[i] are already centered, i.e., have zero mean.

        )EOF");

  subm.def("_subtract_mean", &SubtractMean, py::arg{"values"}.noconvert());

  subm.def("_compute_mean",
      &C ean,
      py values"}.noconvert());
}
}  // namespace detail
}  // namespace netket

namespace netket {

void AddStatsModule(PyObject* m) {
  detail::AddStatsModule(py::module{py::reinterpret_borrow<py::object>(m)});
}

}  // namespace netket
