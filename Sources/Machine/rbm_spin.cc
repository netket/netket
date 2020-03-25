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

#include "rbm_spin.hpp"

#include "Utils/json_utils.hpp"
#include "Utils/messages.hpp"

namespace netket {
/** Initialization of the instance of RbmSpin class (defined in 
  * header) Variables are:
  * @param hilbert - the variable containing information about physical system
  * @param nhidden - number of neurons in hidden layer defined by larger of the values
  *                  (alpha * number of neuron from visible layer) or nhidden
  * @param alpha - density defined as (number of neurons in hidden layer / number of
  *                neurons in visible layer)
  * @param usea - if true use biases in visible layer a*output
  * @param useb - if true use biases in hidden layer b*output
  */
RbmSpin::RbmSpin(std::shared_ptr<const AbstractHilbert> hilbert, int nhidden,
                 int alpha, bool usea, bool useb)
    : AbstractMachine(hilbert), nv_(hilbert->Size()), usea_(usea), useb_(useb) {
  nh_ = std::max(nhidden, alpha * nv_);
  Init();
  
}

/** Member function returning number of visible neurons (inherited from
  * AbstractMachine)
  */
int RbmSpin::Nvisible() const { return nv_; }

/** Member function returning number of parameters (inherited from
  * AbstractMachine)
  */ 
int RbmSpin::Npar() const { return npar_; }

/// Initialization function
void RbmSpin::Init() {
  /** Defining variational parameters: weights and biases (a_ for 
    * visible layer, b_ for hidden layer)
    */
  W_.resize(nv_, nh_);
  a_.resize(nv_);
  b_.resize(nh_);
  /** thetas_ correspond to the "output" of a hidden layer 
    * thetas = W_.transpose() * visible + b_. It is used as the input of cosh
    * functions in the expression for a wavefunction. lnthetas_ correspond to 
    * the ln(cosh(thetas)) which is done by taking the logarithm of a wavefunction
    * (function defined in header)
    */ 
  thetas_.resize(nh_);
  lnthetas_.resize(nh_);
  /** holders for the new thetas and lnthetas values after one or more
    * of the visible units are flipped
    */
  thetasnew_.resize(nh_);
  lnthetasnew_.resize(nh_);

  /** npar_ corresponds to number of variational parameters which, if
    * there are no biases is given by nv_ * nh_ and if there are biases is larger
    * by nv_ and/or nh_.
    */ 
  npar_ = nv_ * nh_;

  if (usea_) {
    npar_ += nv_;
  } else {
    a_.setZero();
  }

  if (useb_) {
    npar_ += nh_;
  } else {
    b_.setZero();
  }
  
  /// Log messages

  InfoMessage() << "RBM Initizialized with nvisible = " << nv_
                << " and nhidden = " << nh_ << std::endl;
  InfoMessage() << "Using visible bias = " << usea_ << std::endl;
  InfoMessage() << "Using hidden bias  = " << useb_ << std::endl;
}

/** Function initializing the parameters of the neural network from
  * random Gaussian distribution (taken from netket library). par variable is used
  * to initialize the values of W_, a_ and b_ by the function SetParameters defined
  * below
  */
void RbmSpin::InitRandomPars(int seed, double sigma) {
  VectorType par(npar_);

  netket::RandomGaussian(par, seed, sigma);

  SetParameters(par);
}

/** Function which initializes a look-up table which can be used to 
  * speed up calculation. It is equal to thetas_ variable - W_.transpose() * v + b_
  * (More information in the abstract_machine.hpp)
  */ 

void RbmSpin::InitLookup(VisibleConstType v, LookupType &lt) {
  /** If look-up table (lt) does not have any vectors then create vector
    * of size nh_
    */ 
  if (lt.VectorSize() == 0) {
    lt.AddVector(b_.size());
  }
  /// If look-up table already have a vector V(0), resize it to nh_
  if (lt.V(0).size() != b_.size()) {
    lt.V(0).resize(b_.size());
  }
  
  /// Definition of look-up table
  lt.V(0) = (W_.transpose() * v + b_);
}

/** Function updating the look-up table where only certain neurons from
  * visible layers have changed. The neurons that changed are specified by &tochange
  * variable and how they changed are represented by &newconf.
  */ 

void RbmSpin::UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                           const std::vector<double> &newconf, LookupType &lt) {
  if (tochange.size() != 0) {
    for (std::size_t s = 0; s < tochange.size(); s++) {
      /// index of changed spin
      const int sf = tochange[s];
      /** adding a value of changed spin to look-up table and removing
        * old value
        */
      lt.V(0) += W_.row(sf) * (newconf[s] - v(sf));
    }
  }
}

/** Function that calculates the derivatives of logarithms of a wavefunction.
  * It is made more efficient by evaluating first the look-up tables.
  */

RbmSpin::VectorType RbmSpin::DerLog(VisibleConstType v) {
  LookupType ltnew;
  InitLookup(v, ltnew);
  return DerLog(v, ltnew);
}

/** Function that calculates the derivatives of logarithms of a wavefunction.
  * Modified version of the code above which uses a look-up table
  * (in this instance both functions are exactly the same)
  */

RbmSpin::VectorType RbmSpin::DerLog(VisibleConstType v, const LookupType &lt) {
  /** VectorType is equivalent to Eigen::Matrix<Complex, Eigen::Dynamic, 1>.
    * der is a vector containing all the derivatives of variational parameters. 
    * der.head correspond to the beginning of the vector, der.segment to the middle
    * and der.tail to the end of the vector
    */
  VectorType der(npar_);
  
  /** derivative of log(wavefunction) with respect to a_ biases is given 
    * by values of visible layers
    */
  if (usea_) {
    der.head(nv_) = v;
  }
  /** derivative of log(wavefunction) with respect to b_ biases is given
    * by tanh(thetas_)
    */
  RbmSpin::tanh(lt.V(0), lnthetas_);

  if (useb_) {
    der.segment(usea_ * nv_, nh_) = lnthetas_;
  }
  /** derivative of log(wavefunction) with respect to W_ weights is given
    * by v * tanh(thetas)
    */ 
  MatrixType wder = (v * lnthetas_.transpose());
  der.tail(nv_ * nh_) = Eigen::Map<VectorType>(wder.data(), nv_ * nh_);

  return der;
}

/// Function that returns parameters stored in W_, a_, b_

RbmSpin::VectorType RbmSpin::GetParameters() {
  VectorType pars(npar_);

  if (usea_) {
    pars.head(nv_) = a_;
  }

  if (useb_) {
    pars.segment(usea_ * nv_, nh_) = b_;
  }

  pars.tail(nv_ * nh_) = Eigen::Map<VectorType>(W_.data(), nv_ * nh_);

  return pars;
}

/// Function that set the parameters pars to the variables W_, a_, b_

void RbmSpin::SetParameters(VectorConstRefType pars) {
  if (usea_) {
    a_ = pars.head(nv_);
  }

  if (useb_) {
    b_ = pars.segment(usea_ * nv_, nh_);
  }

  VectorType Wpars = pars.tail(nv_ * nh_);

  W_ = Eigen::Map<MatrixType>(Wpars.data(), nv_, nh_);
}

// Value of the logarithm of the wave-function
Complex RbmSpin::LogVal(VisibleConstType v) {
  RbmSpin::lncosh(W_.transpose() * v + b_, lnthetas_);

  return (v.dot(a_) + lnthetas_.sum());
}

// Value of the logarithm of the wave-function
// using pre-computed look-up tables for efficiency
Complex RbmSpin::LogVal(VisibleConstType v, const LookupType &lt) {
  RbmSpin::lncosh(lt.V(0), lnthetas_);

  return (v.dot(a_) + lnthetas_.sum());
}

// Difference between logarithms of values, when one or more visible variables
// are being flipped
RbmSpin::VectorType RbmSpin::LogValDiff(
    VisibleConstType v, const std::vector<std::vector<int>> &tochange,
    const std::vector<std::vector<double>> &newconf) {
  /// @param nconn - number of spin flips  
  const std::size_t nconn = tochange.size();
  /** logvaldiffs - stores the difference of logarithms of wavefunctions
    * after and before the flips
    */
  VectorType logvaldiffs = VectorType::Zero(nconn);

  thetas_ = (W_.transpose() * v + b_);
  RbmSpin::lncosh(thetas_, lnthetas_);
  
  /// logtsum contains the sum of logarithms before spin flips
  Complex logtsum = lnthetas_.sum();
  
  /** assigning the values of a logvaldiffs. &tochange is a list of 
    * multiple possible changes to a given configuration. tochange[k] depicts
    * one of those possible changes
    */
  for (std::size_t k = 0; k < nconn; k++) {
    if (tochange[k].size() != 0) {
      thetasnew_ = thetas_;

      for (std::size_t s = 0; s < tochange[k].size(); s++) {
        /** sf is an index of the spin that is currently flipped.
          * newconf[k][s] is a new value for this spin
          */ 
        const int sf = tochange[k][s];
        
        /** this contribution to logvaldiffs(k) comes from the exp(a_ * v)
          * part of the wavefuntion
          */
        logvaldiffs(k) += a_(sf) * (newconf[k][s] - v(sf));

        thetasnew_ += W_.row(sf) * (newconf[k][s] - v(sf));
      }
      /** this contribution to logvaldiffs(k) comes from the cosh(thetas)
        * part of the wavefunction
        */
      RbmSpin::lncosh(thetasnew_, lnthetasnew_);
      logvaldiffs(k) += lnthetasnew_.sum() - logtsum;
    }
  }
  return logvaldiffs;
}

// Difference between logarithms of values, when one or more visible variables
// are being flipped Version using pre-computed look-up tables for efficiency
// on a small number of spin flips

/// Same as above but with look-up tables
Complex RbmSpin::LogValDiff(VisibleConstType v,
                            const std::vector<int> &tochange,
                            const std::vector<double> &newconf,
                            const LookupType &lt) {
  Complex logvaldiff = 0.;

  if (tochange.size() != 0) {
    RbmSpin::lncosh(lt.V(0), lnthetas_);

    thetasnew_ = lt.V(0);

    for (std::size_t s = 0; s < tochange.size(); s++) {
      const int sf = tochange[s];

      logvaldiff += a_(sf) * (newconf[s] - v(sf));

      thetasnew_ += W_.row(sf) * (newconf[s] - v(sf));
    }

    RbmSpin::lncosh(thetasnew_, lnthetasnew_);
    logvaldiff += (lnthetasnew_.sum() - lnthetas_.sum());
  }
  return logvaldiff;
}

/// saving the parameters of the machine

void RbmSpin::Save(const std::string &filename) const {
  json state;
  state["Name"] = "RbmSpin";
  state["Nvisible"] = nv_;
  state["Nhidden"] = nh_;
  state["UseVisibleBias"] = usea_;
  state["UseHiddenBias"] = useb_;
  state["a"] = a_;
  state["b"] = b_;
  state["W"] = W_;
  WriteJsonToFile(state, filename);
}

/// loading the parameters of the machine

void RbmSpin::Load(const std::string &filename) {
  auto const pars = ReadJsonFromFile(filename);
  std::string name = FieldVal<std::string>(pars, "Name");
  if (name != "RbmSpin") {
    throw InvalidInputError(
        "Error while constructing RbmSpin from input parameters");
  }

  if (FieldExists(pars, "Nvisible")) {
    nv_ = FieldVal<int>(pars, "Nvisible");
  }
  if (nv_ != GetHilbert().Size()) {
    throw InvalidInputError(
        "Number of visible units is incompatible with given "
        "Hilbert space");
  }

  if (FieldExists(pars, "Nhidden")) {
    nh_ = FieldVal<int>(pars, "Nhidden");
  } else {
    nh_ = nv_ * double(FieldVal<double>(pars, "Alpha"));
  }

  usea_ = FieldOrDefaultVal(pars, "UseVisibleBias", true);
  useb_ = FieldOrDefaultVal(pars, "UseHiddenBias", true);

  Init();

  /// Loading parameters, if defined in the input
  if (FieldExists(pars, "a")) {
    a_ = FieldVal<VectorType>(pars, "a");
  } else {
    a_.setZero();
  }

  if (FieldExists(pars, "b")) {
    b_ = FieldVal<VectorType>(pars, "b");
  } else {
    b_.setZero();
  }
  if (FieldExists(pars, "W")) {
    W_ = FieldVal<MatrixType>(pars, "W");
  }
}

bool RbmSpin::IsHolomorphic() const noexcept { return true; }

}  // namespace netket
