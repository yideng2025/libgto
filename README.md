# libgto

libgto is a Gaussian integral library for molecular electronic structure. The
current implementation evaluates four-center Coulomb electron-repulsion integrals
for real Cartesian and real solid-harmonic shells through g (`l <= 4`), including
signed and generally contracted coefficients. It provides a C11-compatible API,
a C++17 implementation, reusable prepared pairs, explicit workspaces, and bounded
OpenMP task batches.

Other operators, derivatives, complex/spinor functions and GPU evaluation are not
implemented. The API and ABI are early in development and may change.

## Build and install

Requirements: CMake 3.16 or newer, C11 and C++17 compilers, Boost 1.75 or newer
(headers only), and an OpenMP-capable C++ compiler for parallel batches.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build --parallel
cmake --install build
```

Set `LIBGTO_ENABLE_OPENMP=OFF` for a serial build, or `BUILD_SHARED_LIBS=OFF`
for a static library. If Boost is not in the default search path, provide its
installation prefix using `CMAKE_PREFIX_PATH`. No Python, reference integral
library, BLAS or private code generator is needed for ordinary builds.

CMake clients can use `find_package(libgto CONFIG REQUIRED)` and link
`libgto::gto`. The installed header is `libgto.h`; the shared library is `gto`.
Static clients need a C++ linker and the configured OpenMP runtime.

## Numerical and ownership conventions

The public header defines normalization, angular ordering, contracted-block
layout, error returns and ownership. Coordinates use Bohr. Pair preparation
copies its input arrays; contractions are never implicitly renormalized.
Output buffers belong to the caller; borrowed workspace views have limited
lifetimes. Immutable pairs can be shared, while concurrent evaluations require
separate workspaces or batch handles.

Automatic precision selection retains wide intermediates and original-input
multiprecision reconstruction for difficult inputs. Its risk indicators are
empirical safeguards, not rigorous error certificates for arbitrary cancellation.
The reference policy uses long double and is not an independent exact oracle.
The effective precision of long double depends on the platform. Workspace and
binding limits are local capacities, not a global memory-budget manager.

The standalone Boys API supports orders 0 through 8. Internal ERI auxiliaries
cover the supported shell angular momenta. Optional deterministic table
regeneration uses `libgto_regenerate_boys`; normal builds use checked-in constants.

See LICENSE for the project license.
