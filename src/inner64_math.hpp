#pragma once
#include "internal.hpp"
namespace channel_accumulation {
using gto::Real;
static Real gamma(size_t n, Real eps) {
  Real v = n * eps;
  return v < 1 ? v / (1 - v) : std::numeric_limits<Real>::infinity();
}
// Conditional fixed-G/weight roundoff bound. Angular uncertainty is separate.
static Real bound(Real magnitude, size_t inner, size_t outer, size_t total, double outer_mass) {
  const Real d = std::numeric_limits<double>::epsilon(), r = std::numeric_limits<Real>::epsilon();
  Real upper = magnitude * (1 + gamma(4 * total + 16, r)) * (1 + gamma(4, d));
  Real go = gamma(outer + 2, r);
  Real v = upper * (gamma(inner + 2, d) * (1 + go) + go);
  v += Real(2 * inner + 4) * Real(std::numeric_limits<double>::denorm_min()) * Real(outer_mass) *
       (1 + gamma(4, d));
  return v * (1 + gamma(8, r));
}

static void update(size_t n, double w, const double *in, double *out) {
  for (size_t i = 0; i < n; i++)
    out[i] += w * in[i];
}

static void outer(size_t n, Real w, const double *in, Real *out) {
  for (size_t i = 0; i < n; i++)
    out[i] += w * Real(in[i]);
}
}
