#include "mpi_interface.hpp"

#include <dlfcn.h>
#include <iostream>
#include <memory>

#include "Utils/exceptions.hpp"

namespace netket {

namespace detail {
namespace {
const char* MPIErrorMessage(const int code) {
  constexpr auto buffer_capacity = MPI_MAX_ERROR_STRING + 1;
  static thread_local char buffer[buffer_capacity];
  int buffer_size;
  const auto status = MPI_Error_string(code, buffer, &buffer_size);
  if (status != MPI_SUCCESS) {
    throw InvalidInputError{"invalid error code: " + std::to_string(status)};
  }
  assert(buffer_size < buffer_capacity);
  buffer[buffer_size] = '\0';
  return buffer;
}
}  // namespace
}  // namespace detail

std::string MPIError::make_message(const int status, const char* function) {
  std::ostringstream msg;
  msg << "MPI call to '" << function << "' failed with error code " << status
      << ": " << detail::MPIErrorMessage(status);
  return msg.str();
}

std::string MPIError::make_message(const int status) {
  std::ostringstream msg;
  msg << "MPI failed with error code " << status << ": "
      << detail::MPIErrorMessage(status);
  return msg.str();
}

namespace detail {
namespace {
struct Unload {
  void operator()(void* handle) {
    if (handle) {
      auto const status = dlclose(handle);
      if (status != 0) {
        std::ostringstream msg;
        msg << "dlclosed() failed: " << dlerror() << '\n'
            << "This should not have happened. Please, submit a bug "
               "report to https://github.com/netket/netket/issues.";
        std::cerr << msg.str() << std::endl;
        std::terminate();
      }
    }
  }
};

// See https://github.com/netket/netket/issues/182.
// The workaround is borrowed from
// https://github.com/mpi4py/mpi4py/blob/master/src/lib-mpi/compat/openmpi.h
std::unique_ptr<void, Unload> TryPreload(void) {
  void* handle = nullptr;

#if defined(__linux__) && defined(OMPI_MAJOR_VERSION)
  int mode = RTLD_NOW | RTLD_GLOBAL;
#ifdef RTLD_NOLOAD
  mode |= RTLD_NOLOAD;
#endif
  // NOTE(twesterhout): The following solves the issue on Ubuntu 18.08 with
  // Open MPI 2.1.1
#if OMPI_MAJOR_VERSION == 2
  handle = dlopen("libmpi.so.20", mode);
#endif

  if (!handle) handle = dlopen("libmpi.so", mode);
#endif

  return {handle, Unload{}};
}

struct MPIInitializer {
  MPIInitializer();
  ~MPIInitializer();

 private:
  bool have_initialized_;
};

MPIInitializer::MPIInitializer() {
  int already_initialized;
  MPI_Initialized(&already_initialized);
  if (!already_initialized) {
    auto const handle = TryPreload();
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
}  // namespace

static MPIInitializer initiaze_mpi_when_loading_the_module{};

}  // namespace detail
}  // namespace netket
