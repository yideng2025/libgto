// One-electron quadratic density coupled to an s density, 64 primitive lanes.
#pragma once
#include "internal.hpp"
// Included after the qualified two_p_moment Boys2 implementation.
namespace same_pair_moment {
struct Packet {
  double a[3][64], b[3][64], v[3][64], cov0[64], cov1[64], F[3][64], G[9][64], T[64], pref[64],
      budget[64];
  bool valid[64];
};
template <int I, int J> static void pp(Packet &p, size_t n) {
  for (size_t k = 0; k < n; k++) {
    double q0 = p.a[I][k] * p.b[J][k], q1 = p.a[I][k] * p.v[J][k] + p.v[I][k] * p.b[J][k],
           q2 = p.v[I][k] * p.v[J][k];
    if constexpr (I == J) {
      q0 += p.cov0[k];
      q1 += p.cov1[k];
    }
    p.G[I + 3 * J][k] = p.pref[k] * ((q0 * p.F[0][k] + q1 * p.F[1][k]) + q2 * p.F[2][k]);
  }
}
template <int M> static void d(Packet &p, size_t n) {
  constexpr double rt = 1.7320508075688772935274463415058723669;
  for (size_t k = 0; k < n; k++) {
    double ax = p.a[0][k], ay = p.a[1][k], az = p.a[2][k], vx = p.v[0][k], vy = p.v[1][k],
           vz = p.v[2][k], q0, q1, q2;
    if constexpr (M == 0) {
      q0 = rt * ax * ay;
      q1 = rt * (ax * vy + ay * vx);
      q2 = rt * vx * vy;
    }
    if constexpr (M == 1) {
      q0 = rt * ay * az;
      q1 = rt * (ay * vz + az * vy);
      q2 = rt * vy * vz;
    }
    if constexpr (M == 2) {
      q0 = az * az - .5 * (ax * ax + ay * ay);
      q1 = 2 * az * vz - (ax * vx + ay * vy);
      q2 = vz * vz - .5 * (vx * vx + vy * vy);
    }
    if constexpr (M == 3) {
      q0 = rt * ax * az;
      q1 = rt * (ax * vz + az * vx);
      q2 = rt * vx * vz;
    }
    if constexpr (M == 4) {
      q0 = .5 * rt * (ax * ax - ay * ay);
      q1 = rt * (ax * vx - ay * vy);
      q2 = .5 * rt * (vx * vx - vy * vy);
    }
    p.G[M][k] = p.pref[k] * ((q0 * p.F[0][k] + q1 * p.F[1][k]) + q2 * p.F[2][k]);
  }
}
static void angular(Packet &p, size_t n, bool single_d) {
  for (size_t k = 0; k < n; k++) {
    double f[3]{};
    if (p.valid[k])
      two_p_moment::boys(p.T[k], f);
    for (int m = 0; m < 3; m++)
      p.F[m][k] = f[m];
  }
  if (single_d) {
    d<0>(p, n);
    d<1>(p, n);
    d<2>(p, n);
    d<3>(p, n);
    d<4>(p, n);
  } else {
    pp<0, 0>(p, n);
    pp<1, 0>(p, n);
    pp<2, 0>(p, n);
    pp<0, 1>(p, n);
    pp<1, 1>(p, n);
    pp<2, 1>(p, n);
    pp<0, 2>(p, n);
    pp<1, 2>(p, n);
    pp<2, 2>(p, n);
  }
  constexpr double eps = std::numeric_limits<double>::epsilon();
  for (size_t k = 0; k < n; k++) {
    double ma = 0, mb = 0;
    for (int axis = 0; axis < 3; axis++) {
      ma = std::max(ma, std::abs(p.a[axis][k]) + std::abs(p.v[axis][k]));
      mb = std::max(mb, std::abs(p.b[axis][k]) + std::abs(p.v[axis][k]));
    }
    double h = single_d ? 2 * ma * ma : ma * mb + std::abs(p.cov0[k]) + std::abs(p.cov1[k]);
    double bound = std::abs(p.pref[k]) * p.F[0][k] * h * (2048 * eps * (1 + std::min(p.T[k], 2.5)));
    bound = bound * (1 + 64 * eps) + std::numeric_limits<double>::min();
    p.valid[k] &= std::isfinite(bound) && bound <= p.budget[k];
    for (int v = 0; v < (single_d ? 5 : 9); v++)
      if (!std::isfinite(p.G[v][k]) || (p.G[v][k] != 0 && !std::isnormal(p.G[v][k])))
        p.valid[k] = false;
  }
}
}
