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

#ifndef NETKET_ABSTRACTHILBERT_HPP
#define NETKET_ABSTRACTHILBERT_HPP

#include <Eigen/Core>
#include <complex>
#include <limits>
#include <nonstd/optional.hpp>
#include <vector>

#include "Graph/abstract_graph.hpp"
#include "Hilbert/hilbert_index.hpp"
#include "Utils/random_utils.hpp"

namespace netket {

/**
  Abstract class for Hilbert spaces.
  This class prototypes the methods needed
  by a class satisfying the Hilbert concept.
*/
class AbstractHilbert {
 public:
  /**
  Member function returning true if the hilbert space has discrete quantum
  numbers.
  @return true if the local hilbert space is discrete
  */
  virtual bool IsDiscrete() const = 0;

  /**
   * We call a Hilbert space indexable if and only if the total Hilbert space
   * dimension can be represented by an index of type int.
   * @return true, iff the Hilbert space is indexable
   */
  bool IsIndexable() const {
    static constexpr int max_states = std::numeric_limits<int>::max() - 1;
    static const double log_max = std::log(max_states);

    return Size() * std::log(LocalSize()) <= log_max;
  }

  const HilbertIndex &GetIndex() const {
    if (!index_.has_value()) {
      if (!IsIndexable()) {
        throw std::runtime_error("Hilbert space is too large to be indexed");
      }
      index_.emplace(LocalStates(), LocalSize(), Size());
    }
    return *index_;
  }

  /**
  Member function returning the size of the local hilbert space.
  @return Size of the discrete local hilbert space. For continous spaces an
  error message is returned.
  */
  virtual int LocalSize() const = 0;

  /**
  Member function returning the number of visible units needed to describe the
  system.
  @return Number of visible units needed to described the system.
  */
  virtual int Size() const = 0;

  /**
  Member function returning the local states.
  @return Vector containing the value of the discrete local quantum numbers. If
  the local quantum numbers are continous, the vector contains lower and higher
  bounds for the local quantum numbers.
  */
  virtual std::vector<double> LocalStates() const = 0;

  /**
  Member function generating uniformely distributed local random states
  @param state a reference to a visible configuration, in output this contains
  the random state.
  @param rgen the random number generator to be used
  */
  virtual void RandomVals(Eigen::Ref<Eigen::VectorXd> state,
                          netket::default_random_engine &rgen) const = 0;

  /**
  Member function updating a visible configuration using the information on
  where the local changes have been done.
  @param v is the vector visible units to be modified.
  @param tochange contains a list of which quantum numbers are to be modified.
  @param newconf contains the value that those quantum numbers should take
  */
  virtual void UpdateConf(Eigen::Ref<Eigen::VectorXd> v,
                          const std::vector<int> &tochange,
                          const std::vector<double> &newconf) const = 0;

  virtual const AbstractGraph &GetGraph() const noexcept = 0;

  // Allow range-based for over all states in the Hilbert space, iff it is
  // indexable
  StateGenerator begin() const { return StateGenerator(GetIndex()); };
  StateGenerator end() const { return StateGenerator(GetIndex()); };

  virtual ~AbstractHilbert() = default;

 private:
  mutable nonstd::optional<HilbertIndex> index_;
};

}  // namespace netket

#endif
