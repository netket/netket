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

#include "netket.hpp"
#include <mpi.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  netket::Welcome(argc);

  try {
    auto pars = netket::ReadJsonFromFile(argv[1]);

    // DEPRECATED (to remove for v2.0.0)
    if (netket::FieldExists(pars, "GroundState") ||
        netket::FieldExists(pars, "Learning")) {
      netket::GroundState gs(pars);

    } else if (netket::FieldExists(pars, "TimeEvolution")) {
      netket::RunTimeEvolution(pars);

    } else if (netket::FieldExists(pars, "Supervised")) {
      netket::ErrorMessage()
          << "Supervised Learning still under development, try later."
          << "\n";

    } else if (netket::FieldExists(pars, "Unsupervised")) {
      netket::Unsupervised unsup(pars);
    } else {
      netket::ErrorMessage()
          << "No task specified. Please include one of the sections"
             "'GroundState' or 'TimeEvolution' in the input file.\n";
    }

  } catch (const netket::InvalidInputError &e) {
    netket::ErrorMessage() << e.what() << "\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;
}
