#pragma once
#include <cstdint>
namespace gto {
void angular_products_ordered(int, int, int, int, const double *, const double *, const double *,
                              double *, double *);
void angular_products_block4(int, int, int, int, const double *, const double *, const double *,
                             double *, double *);
void angular_products(int, int, int, int, const double *, const double *, const double *, double *,
                      double *);
void angular_bra_ordered(int, int, int, const double *, const double *, double *);
bool angular_unique_R(int, int, int, int, const unsigned short *, const double *, const double *,
                      const double *, double *, double *);
// Signed unique R is [positive nr][negative nr], using the Hermite coupling codes.
void angular_ps_dd_bra_unique(const unsigned short *, const double *, const double *,
                              const double *, double *, double *);
void angular_dd_ps_ket_unique(const unsigned short *, const double *, const double *,
                              const double *, double *, double *);
void angular_ds_bra_unique(int, const unsigned short *, const double *, const double *,
                           const double *, const std::uint64_t *, double *, double *);
}
