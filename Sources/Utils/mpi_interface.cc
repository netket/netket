#include "mpi_interface.hpp"

#include <iostream>

namespace netket {
namespace detail {
MPIInitializer::MPIInitializer() {
  int already_initialized;
  MPI_Initialized(&already_initialized);
  if (!already_initialized) {
    // We don't have access to command-line arguments
    if (MPI_Init(nullptr, nullptr) != MPI_SUCCESS) {
      std::ostringstream msg;
      msg << "This should never have happened. How did you manage to "
             "call MPI_Init() in between two C function calls?! "
             "Terminating now.";
      std::cerr << msg.str() << std::endl;
      std::terminate();
    }
    have_initialized_ = true;
#if !defined(NDEBUG)
    std::cerr << "MPI successfully initialized by NetKet." << std::endl;
#endif
  }
}

MPIInitializer::~MPIInitializer() {
  if (have_initialized_) {
    // We have initialized MPI so it's only right we finalize it.
    MPI_Finalize();
#if !defined(NDEBUG)
    std::cerr << "MPI successfully finalized by NetKet." << std::endl;
#endif
  }
}

}  // namespace detail
}  // namespace netket
