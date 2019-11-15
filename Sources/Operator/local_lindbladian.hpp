//
// Created by Filippo Vicentini on 08/11/2019.
//

#ifndef NETKET_LOCAL_LINDBLADIAN_HPP
#define NETKET_LOCAL_LINDBLADIAN_HPP

#include <utility>

#include "Graph/doubled_graph.hpp"
#include "Hilbert/abstract_hilbert.hpp"
#include "Hilbert/doubled_hilbert.hpp"
#include "Operator/abstract_operator.hpp"
#include "Operator/local_operator.hpp"

namespace netket {
class LocalLindbladian : public AbstractOperator {
 public:
  using MelType = std::vector<Complex>;
  using MatType = std::vector<std::vector<MelType>>;
  using SiteType = std::vector<int>;
  using MapType = std::map<std::vector<double>, int>;
  using StateType = std::vector<std::vector<double>>;
  using ConnType = std::vector<std::vector<int>>;
  using VectorType = AbstractOperator::VectorType;
  using VectorRefType = AbstractOperator::VectorRefType;
  using VectorConstRefType = AbstractOperator::VectorConstRefType;

 private:
  LocalOperator Hnh_;
  LocalOperator H_;
  std::vector<LocalOperator> jump_ops_;

  LocalOperator Hnh_dag_;

 public:
  explicit LocalLindbladian(const LocalOperator &H);

  void Init();
  const std::vector<LocalOperator> &GetJumpOperators() const;

  void AddJumpOperator(const LocalOperator &op);

  void FindConn(VectorConstRefType v, std::vector<Complex> &mel,
                std::vector<std::vector<int>> &connectors,
                std::vector<std::vector<double>> &newconfs) const override;

  void FindConnSuperOp(VectorConstRefType vrow, VectorConstRefType vcol,
                       std::vector<Complex> &mel,
                       std::vector<std::vector<int>> &connectors,
                       std::vector<std::vector<double>> &newconfs) const;

  const DoubledHilbert &GetHilbertDoubled() const;
  const LocalOperator& GetEffectiveHamiltonian() const {
    return Hnh_;
  };

  std::shared_ptr<const DoubledHilbert> GetHilbertDoubledShared() const;
};
}  // namespace netket

#endif  // NETKET_LOCAL_LINDBLADIAN_HPP
