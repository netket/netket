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

#ifndef NETKET_WELCOME_HPP
#define NETKET_WELCOME_HPP

#include <mpi.h>
#include <iostream>

namespace netket {

void Welcome(int argc) {
  int mynode;

  MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
  if (mynode == 0) {
    std::cout << "############################################" << std::endl;
    std::cout << "# NetKet version 1.0.3                     #" << std::endl;
    std::cout << "# Website: https://www.netket.org          #" << std::endl;
    std::cout << "# Licensed under Apache-2.0 - see LICENSE  #" << std::endl;
    std::cout << "############################################" << std::endl;
    std::cout << std::endl << std::endl;
  }

  if (argc != 2) {
    if (mynode == 0) {
      std::cout << "# Usage: Insert name of input Json file" << std::endl;
      std::cout << std::endl << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    std::exit(0);
  }
}
}  // namespace netket
#endif
