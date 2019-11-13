//
// Created by Filippo Vicentini on 08/11/2019.
//

#include "local_lindbladian.hpp"
#include "abstract_operator.hpp"

namespace netket {

LocalLindbladian::LocalLindbladian(const LocalOperator &H)
    : AbstractOperator(std::make_shared<DoubledHilbert>(H.GetHilbertShared())),
      Hnh_(H) {
  Init();
}

void LocalLindbladian::Init() {
  auto j = std::complex<double>(0.0, 1.0);

  for (auto &L : jump_ops_) {
    Hnh_ += -0.5 * j * L.Transpose().Conjugate() * L;
  }
}

const std::vector<const LocalOperator> &LocalLindbladian::GetJumpOperators()
    const {
  return jump_ops_;
}

void LocalLindbladian::AddJumpOperator(const netket::LocalOperator &op) {
  jump_ops_.push_back(op);

  Hnh_ += std::complex<double>(0.0, 1.0) * op.Transpose().Conjugate() * op;
}

void LocalLindbladian::FindConn(
    VectorConstRefType v, std::vector<Complex> &mel,
    std::vector<std::vector<int>> &connectors,
    std::vector<std::vector<double>> &newconfs) const {
  auto im = Complex(0.0, 1.0);
  auto N = GetHilbertDoubled().SizePhysical();
  auto vrow = v.head(N);
  auto vcol = v.tail(N);

  Hnh_.ForEachConn(vrow, [&](ConnectorRef conn) {
    mel.push_back(-im * conn.mel);

    // TODO  Extremely ugly, but alternatives don't work (why?!)
    auto conn_tmp = std::vector<int>(
        conn.tochange.data(), conn.tochange.data() + conn.tochange.size());
    auto newconf_tmp = std::vector<double>(
        conn.newconf.data(), conn.newconf.data() + conn.newconf.size());

    // Those are changes for the rows...

    connectors.push_back(conn_tmp);
    newconfs.push_back(newconf_tmp);
  });

  Hnh_.ForEachConn(vcol, [&](ConnectorRef conn) {
    mel.push_back(im * std::conj(conn.mel));

    // TODO  Extremely ugly, but alternatives don't work (why?!)
    auto conn_tmp = std::vector<int>(
        conn.tochange.data(), conn.tochange.data() + conn.tochange.size());
    auto newconf_tmp = std::vector<double>(
        conn.newconf.data(), conn.newconf.data() + conn.newconf.size());

    // Those are changes for the columns: add number of sites...
    std::transform(conn_tmp.begin(), conn_tmp.end(), conn_tmp.begin(),
                   bind2nd(std::plus<double>(), N));

    connectors.push_back(conn_tmp);
    newconfs.push_back(newconf_tmp);
  });

  for (auto &op : jump_ops_) {
    op.ForEachConn(vrow, [&](ConnectorRef conn_row) {
      auto conn_tmp_row =
          std::vector<int>(conn_row.tochange.data(),
                           conn_row.tochange.data() + conn_row.tochange.size());
      auto newconf_tmp_row = std::vector<double>(
          conn_row.newconf.data(),
          conn_row.newconf.data() + conn_row.newconf.size());

      op.ForEachConn(vcol, [&](ConnectorRef conn_col) {
        auto conn_tmp_col = std::vector<int>(
            conn_col.tochange.data(),
            conn_col.tochange.data() + conn_col.tochange.size());
        auto newconf_tmp_col = std::vector<double>(
            conn_col.newconf.data(),
            conn_col.newconf.data() + conn_col.newconf.size());

        std::transform(conn_tmp_col.begin(), conn_tmp_col.end(),
                       conn_tmp_col.begin(), bind2nd(std::plus<double>(), N));

        mel.push_back(conn_row.mel * std::conj(conn_col.mel));
        connectors.push_back(conn_tmp_row);
        newconfs.push_back(newconf_tmp_row);

        connectors.back().insert(connectors.back().end(), conn_tmp_col.begin(),
                                 conn_tmp_col.end());
        newconfs.back().insert(newconfs.back().end(), newconf_tmp_col.begin(),
                               newconf_tmp_col.end());
      });
    });
  }
}

void LocalLindbladian::FindConnSuperOp(
    VectorConstRefType vrow, VectorConstRefType vcol, std::vector<Complex> &mel,
    std::vector<std::vector<int>> &connectors,
    std::vector<std::vector<double>> &newconfs) const {
  Hnh_.FindConn(vrow, mel, connectors, newconfs);
  for (auto &op : jump_ops_) {
    op.FindConn(vrow, mel, connectors, newconfs, false);
    op.FindConn(vcol, mel, connectors, newconfs, false);
  }
}

const DoubledHilbert &LocalLindbladian::GetHilbertDoubled() const {
  return *GetHilbertDoubledShared();
}

std::shared_ptr<const DoubledHilbert>
LocalLindbladian::GetHilbertDoubledShared() const {
  return std::static_pointer_cast<const DoubledHilbert>(GetHilbertShared());
}

}  // namespace netket