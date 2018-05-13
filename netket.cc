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

#include "netket.hh"
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  if (argc != 2) {
    std::cerr << "Insert name of input Json file" << std::endl;
    std::abort();
  }

  auto pars = netket::ReadJsonFromFile(argv[1]);

  netket::Learning learning(pars);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
