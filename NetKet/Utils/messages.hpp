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

#ifndef NETKET_MESSAGES_HPP
#define NETKET_MESSAGES_HPP

#include <mpi.h>
#include <string>

namespace netket {

void InfoMessage(const std::string & message) {
  int mynode;
  MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
  if (mynode == 0) {
    std::cout << "# " << message << std::endl;
  }
}

void WarningMessage(const std::string & message) {
  int mynode;
  MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
  if (mynode == 0) {
    std::cerr << "# WARNING" << std::endl;
    std::cerr << "# " << message << std::endl;
  }
}

void ErrorMessage(const std::string & message) {
  int mynode;
  MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
  if (mynode == 0) {
    std::cerr << "# ERROR" << std::endl;
    std::cerr << "# " << message << std::endl;
  }
}

} // namespace netket

#endif // NETKET_MESSAGES_HPP
