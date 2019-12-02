//
// Created by Filippo Vicentini on 08/11/2019.
//

#ifndef NETKET_LOCAL_LIOUVILLIAN_HPP
#define NETKET_LOCAL_LIOUVILLIAN_HPP

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
      Class describing the Liouvillian of a Lindblad Master Equation for
      Open Quantum systems
*/
class LocalLiouvillian : public AbstractOperator {
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
  // The effective non hermitian hamiltonian Hnh_ = H_ - im/2\sum_i L_i^dag L_i
  LocalOperator Hnh_;
  LocalOperator H_;
  std::vector<LocalOperator> jump_ops_;

  // Adjoint of Hnh_
  LocalOperator Hnh_dag_;

 public:
  /**
   * Constructs a LocalLiouvillian from the hamiltonian H and with no jump
   * operators. The Hamiltonian must be a LocalOperator (other types of
   * operators are not supported).
   * @param H : the Hamiltonian
   */
  explicit LocalLiouvillian(const LocalOperator &H);

  /**
   * Member function to construct the effective non-hermitian hamiltonian Hnh_
   * and Hnh_dag_ starting from the Hamiltonian and the jump_operators.
   */
  void Init();

  /**
   * Member function to access the list of jump operators associated with the
   * dissipative part of this Liouvillian.
   * @return a vector of LocalOperators.
   */
  const std::vector<LocalOperator> &GetJumpOperators() const;

  /**
   * Member function to add another jump operator to the dissipative part of
   * this liouvillian.
   * @param op : LocalOperator taken as a constant reference.
   */
  void AddJumpOperator(const LocalOperator &op);

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
   * Member function returning the doubled hilbert space associated with this
   * Liouvillian. It returns the same object as GetHilbert(), but performs a
   * static_cast to convert it to the right super-class.
   * @return Dobuled Hilbert space
  */
  const DoubledHilbert &GetHilbertDoubled() const;
  const LocalOperator &GetEffectiveHamiltonian() const { return Hnh_; };

  /**
   * Member function returning the doubled hilbert space associated with this
   * Lindbladian. It returns the same object as GetHilbertShared(), but performs
   * a static_cast to convert it to the right super-class.
   * @return shared_ptr to the doubled Hilbert space
   */
  std::shared_ptr<const DoubledHilbert> GetHilbertDoubledShared() const;

  // Members below are inherited from AbstractOperator.

  void FindConn(VectorConstRefType v, MelType &mel, ConnectorsType &connectors,
                NewconfsType &newconfs) const override;

  void ForEachConn(VectorConstRefType v, ConnCallback callback) const override;


};
}  // namespace netket

#endif  // NETKET_LOCAL_LIOUVILLIAN_HPP
