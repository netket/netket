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

#ifndef NETKET_MACHINE_HPP
#define NETKET_MACHINE_HPP

#include <fstream>
#include <memory>

#include "Graph/graph.hpp"
#include "Operator/operator.hpp"
#include "abstract_machine.hpp"
#include "ffnn.hpp"
#include "jastrow.hpp"
#include "jastrow_symm.hpp"
#include "mpark/variant.hpp"
#include "mps_periodic.hpp"
#include "rbm_multival.hpp"
#include "rbm_spin.hpp"
#include "rbm_spin_symm.hpp"

namespace netket {

template <typename T>
class Machine : public AbstractMachine<T> {
 public:
  using VariantType =
      mpark::variant<RbmSpin<T>, RbmSpinSymm<T>, FFNN<T>, Jastrow<T>,
                     JastrowSymm<T>, RbmMultival<T>, MPSPeriodic<T, true>,
                     MPSPeriodic<T, false>>;
  using VectorType = typename AbstractMachine<T>::VectorType;
  using MatrixType = typename AbstractMachine<T>::MatrixType;
  using StateType = typename AbstractMachine<T>::StateType;
  using LookupType = typename AbstractMachine<T>::LookupType;
  using VectorRefType = typename AbstractMachine<T>::VectorRefType;
  using VectorConstRefType = typename AbstractMachine<T>::VectorConstRefType;
  using VisibleConstType = typename AbstractMachine<T>::VisibleConstType;

 private:
  VariantType obj_;

 public:
  Machine(VariantType obj) : obj_(std::move(obj)) {}

  /**
  Member function returning the number of variational parameters.
  @return Number of variational parameters in the Machine.
  */
  int Npar() const override {
    return mpark::visit([](auto &&obj) { return obj.Npar(); }, obj_);
  }
  /**
  Member function returning the current set of parameters in the machine.
  @return Current set of variational parameters of the machine.
  */
  VectorType GetParameters() override {
    return mpark::visit([](auto &&obj) { return obj.GetParameters(); }, obj_);
  }

  /**
  Member function setting the current set of parameters in the machine.
  */
  void SetParameters(VectorConstRefType pars) override {
    mpark::visit([=](auto &&obj) { obj.SetParameters(pars); }, obj_);
  }

  /**
  Member function providing a random initialization of the parameters.
  @param seed is the see of the random number generator.
  @param sigma is the variance of the gaussian.
  */
  void InitRandomPars(int seed, double sigma) override {
    mpark::visit([=](auto &&obj) { obj.InitRandomPars(seed, sigma); }, obj_);
  }

  /**
  Member function returning the number of visible units.
  @return Number of visible units in the Machine.
  */
  int Nvisible() const override {
    return mpark::visit([](auto &&obj) { return obj.Nvisible(); }, obj_);
  }

  /**
  Member function computing the logarithm of the wave function for a given
  visible vector. Given the current set of parameters, this function should
  comput the value of the logarithm of the wave function from scratch.
  @param v a constant reference to a visible configuration.
  @return Logarithm of the wave function.
  */
  T LogVal(VisibleConstType v) override {
    return mpark::visit([=](auto &&obj) { return obj.LogVal(v); }, obj_);
  }

  /**
  Member function computing the logarithm of the wave function for a given
  visible vector. Given the current set of parameters, this function should
  comput the value of the logarithm of the wave function using the information
  provided in the look-up table, to speed up the computation.
  @param v a constant reference to a visible configuration.
  @param lt a constant eference to the look-up table.
  @return Logarithm of the wave function.
  */
  T LogVal(VisibleConstType v, const LookupType &lt) override {
    return mpark::visit([v, &lt](auto &&obj) { return obj.LogVal(v, lt); },
                        obj_);
  }

  /**
  Member function initializing the look-up tables.
  If needed, a Machine can make use of look-up tables
  to speed up some critical functions. For example,
  to speed up the calculation of wave-function ratios.
  The state of a look-up table depends on the visible units.
  This function should initialize the look-up tables
  making sure that memory in the table is also being allocated.
  @param v a constant reference to the visible configuration.
  @param lt a reference to the look-up table to be initialized.
  */
  void InitLookup(VisibleConstType v, LookupType &lt) override {
    mpark::visit([v, &lt](auto &&obj) { obj.InitLookup(v, lt); }, obj_);
  }

  /**
  Member function updating the look-up tables.
  If needed, a Machine can make use of look-up tables
  to speed up some critical functions. For example,
  to speed up the calculation of wave-function ratios.
  The state of a look-up table depends on the visible units.
  This function should update the look-up tables
  when the state of visible units is changed according
  to the information stored in toflip and newconf
  @param v a constant reference to the current visible configuration.
  @param tochange a constant reference to a vector containing the indeces of the
  units to be modified.
  @param newconf a constant reference to a vector containing the new values of
  the visible units: here newconf(i)=v'(tochange(i)), where v' is the new
  visible state.
  @param lt a reference to the look-up table to be updated.
  */
  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override {
    mpark::visit(
        [v, &tochange, &newconf, &lt](auto &&obj) {
          return obj.UpdateLookup(v, tochange, newconf, lt);
        },
        obj_);
  }

  /**
  Member function computing the difference between the logarithm of the
  wave-function computed at different values of the visible units (v, and a set
  of v').
  @param v a constant reference to the current visible configuration.
  @param tochange a constant reference to a vector containing the indeces of the
  units to be modified.
  @param newconf a constant reference to a vector containing the new values of
  the visible units: here for each v', newconf(i)=v'(tochange(i)), where v' is
  the new visible state.
  @return A vector containing, for each v', log(Psi(v')) - log(Psi(v))
  */
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override {
    return mpark::visit(
        [v, &tochange, &newconf](auto &&obj) {
          return obj.LogValDiff(v, tochange, newconf);
        },
        obj_);
  }

  /**
  Member function computing the difference between the logarithm of the
  wave-function computed at different values of the visible units (v, and a
  single v'). This version uses the look-up tables to speed-up the calculation.
  @param v a constant reference to the current visible configuration.
  @param tochange a constant reference to a vector containing the indeces of the
  units to be modified.
  @param newconf a constant reference to a vector containing the new values of
  the visible units: here newconf(i)=v'(tochange(i)), where v' is the new
  visible state.
  @param lt a constant eference to the look-up table.
  @return The value of log(Psi(v')) - log(Psi(v))
  */
  virtual T LogValDiff(VisibleConstType v, const std::vector<int> &toflip,
                       const std::vector<double> &newconf,
                       const LookupType &lt) override {
    return mpark::visit(
        [v, &toflip, &newconf, &lt](auto &&obj) {
          return obj.LogValDiff(v, toflip, newconf, lt);
        },
        obj_);
  }

  /**
  Member function computing the derivative of the logarithm of the wave function
  for a given visible vector.
  @param v a constant reference to a visible configuration.
  @return Derivatives of the logarithm of the wave function with respect to the
  set of parameters.
  */
  VectorType DerLog(VisibleConstType v) override {
    return mpark::visit([=](auto &&obj) { return obj.DerLog(v); }, obj_);
  }

  void to_json(json &j) const override {
    mpark::visit([&j](auto &&obj) { obj.to_json(j); }, obj_);
  }
  virtual void from_json(const json &j) override {
    mpark::visit([&j](auto &&obj) { obj.from_json(j); }, obj_);
  }

  Hilbert GetHilbert() const override {
    return mpark::visit([](auto &&obj) { return obj.GetHilbert(); }, obj_);
  }
};

}  // namespace netket
#endif
