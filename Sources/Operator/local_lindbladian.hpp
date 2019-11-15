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
/**
 * This struct represents a non-zero matrix-element L(v,v_row', v_col') of a
 * super-operator for a given visible state v=(v_row, v_col).
 */
struct ConnectorSuperopRef {
  /// The matrix element L(v_row, v_col ,v_row', v_col')
  Complex mel;
  /// The indices at which v_row needs to be changed to obtain v_row'
  nonstd::span<const int> tochange_row;
  /// The new values such that
  ///    v'(tochange_row[k]) = newconf_row[k]
  /// and v'(i) = v(i) for i ∉ tochange_row.
  nonstd::span<const double> newconf_row;

  /// The indices at which v_col needs to be changed to obtain v_col'
  nonstd::span<const int> tochange_col;
  nonstd::span<const double> newconf_col;
};

/**
      Class describing a Lindblad Master Equation for Open Quantum systems
*/
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
  using ConnSuperOpCallback = std::function<void(ConnectorSuperopRef)>;

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

  void FindConn(VectorConstRefType v, MelType &mel, ConnectorsType &connectors,
                NewconfsType &newconfs) const override;

  void ForEachConn(VectorConstRefType v, ConnCallback callback) const override;

  /**
   * Iterates over all states reachable from a given visible configuration
   * v=(vr, vc), i.e., all states v'=(vr',vc') such that O(vr,vc,vr',vc') is
   * non-zero.
   * @param vr The row configuration.
   * @param vc The column configuration.
   * @param callback Function void callback(ConnSuperOpCallback conn) which will
   * be called once for each reachable configuration v'. The parameter conn
   * contains the value O(v,v') and the information to obtain vr' and vc' from
   * vr and vc.
   */
  void ForEachConnSuperOp(VectorConstRefType vrow, VectorConstRefType vcol,
                          ConnSuperOpCallback callback) const;

  /**
  Member function returning the doubled hilbert space associated with this
  Lindbladian. It returns the same object as GetHilbert(), but performs a
  static_cast to convert it to the right super-class.
  @return Dobuled Hilbert space specifier for this Hamiltonian
  */
  const DoubledHilbert &GetHilbertDoubled() const;
  const LocalOperator &GetEffectiveHamiltonian() const { return Hnh_; };

  std::shared_ptr<const DoubledHilbert> GetHilbertDoubledShared() const;
};
}  // namespace netket

#endif  // NETKET_LOCAL_LINDBLADIAN_HPP
