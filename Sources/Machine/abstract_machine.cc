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

#include "abstract_machine.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

namespace netket {

AbstractMachine::VectorType AbstractMachine::DerLogChanged(
    VisibleConstType v, const std::vector<int> &tochange,
    const std::vector<double> &newconf) {
  VisibleType vp(v);
  hilbert_->UpdateConf(vp, tochange, newconf);
  return DerLog(vp);
}

void AbstractMachine::Save(const std::string &filename) const {
  std::ofstream filewf(filename);
  Save(filewf);
  filewf.close();
}

void AbstractMachine::Save(std::ofstream &stream) const {
  nlohmann::json j;
  to_json(j);
  stream << j << std::endl;
}

}  // namespace netket
