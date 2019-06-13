
#ifndef NETKET_MACHINE_PY_ABSTRACT_MATHINE_HPP
#define NETKET_MACHINE_PY_ABSTRACT_MATHINE_HPP

#include "Machine/abstract_machine.hpp"

namespace netket {

class PyAbstractMachine : public AbstractMachine {
 public:
  PyAbstractMachine(std::shared_ptr<const AbstractHilbert> hilbert)
      : AbstractMachine{std::move(hilbert)} {}

  int Npar() const override;
  int Nvisible() const override;
  bool IsHolomorphic() const noexcept override;

  VectorType GetParameters() override;
  void SetParameters(VectorConstRefType pars) override;
  void InitRandomPars(int const seed, double const sigma) override;

  Complex LogVal(VisibleConstType v) override;
  Complex LogVal(VisibleConstType v, const LookupType & /*unused*/) override;

  void InitLookup(VisibleConstType /*unused*/,
                  LookupType & /*unused*/) override;
  void UpdateLookup(VisibleConstType /*unused*/,
                    const std::vector<int> & /*unused*/,
                    const std::vector<double> & /*unused*/,
                    LookupType & /*unused*/) override;

  VectorType LogValDiff(
      VisibleConstType old_v, const std::vector<std::vector<int>> &to_change,
      const std::vector<std::vector<double>> &new_conf) override;
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &to_change,
                     const std::vector<double> &new_conf,
                     const LookupType & /*unused*/) override;

  VectorType DerLog(VisibleConstType v) override;
  VectorType DerLog(VisibleConstType v, const LookupType & /*lt*/) override;
  VectorType DerLogChanged(VisibleConstType old_v,
                           const std::vector<int> &to_change,
                           const std::vector<double> &new_conf) override;

  void Save(const std::string &filename) const override;
  void Load(const std::string &filename) override;

  ~PyAbstractMachine() override = default;

 private:
  inline Complex LogValDiff(VisibleConstType old_v,
                            const std::vector<int> &to_change,
                            const std::vector<double> &new_conf);
};

}  // namespace netket

#endif  // NETKET_MACHINE_PY_ABSTRACT_MATHINE_HPP
