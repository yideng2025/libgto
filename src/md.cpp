#include <cfenv>
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif
#include "internal.hpp"
#include "r_dependency.hpp"
#include "boys_table.hpp"
#include "angular_products.hpp"
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "single_p_moment_core.hpp"
#include "two_p_core.hpp"
#include "two_p_moment_polybound.hpp"
#include "dplusp_core_boys_shared.hpp"
#include "same_side_core.hpp"
#include "three_p_moment_symmetric_policy.hpp"
#include "inner64_math.hpp"
#include "fundamental4_policy.hpp"
#include "dd_moment_core.hpp"

namespace gto {
template <bool Uniform>
static inline void update_channels(size_t n, double weight, const double *values,
                                   const double *errors, double *out, double *error_out) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  double angular = 0;
  if constexpr (Uniform)
    angular = std::abs(weight) * errors[0];
  for (size_t v = 0; v < n; v++) {
    double product = weight * values[v];
    out[v] += product;
    double e;
    if constexpr (Uniform)
      e = angular;
    else
      e = std::abs(weight) * errors[v];
    error_out[v] += e + 2 * eps * std::abs(product) + eps * std::abs(out[v]);
  }
}

static bool ordered_environment() {
  if (std::fegetround() != FE_TONEAREST)
    return false;
#if defined(__x86_64__)
  if (_mm_getcsr() & ((1u << 15) | (1u << 6)))
    return false;
#endif
  return true;
}

static unsigned first_set_bit(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
  return unsigned(__builtin_ctzll(v));
#else
  unsigned n = 0;
  while (!(v & 1)) {
    v >>= 1;
    ++n;
  }
  return n;
#endif
}

// McMurchie and Davidson, J. Comput. Phys. 26 (1978), 218-231,
// doi:10.1016/0021-9991(78)90092-X. Direct scalar Hermite expansion;

// Boys constants are reproduced by the included C++17 generator.
std::vector<Triple> orbitals(int L) {
  std::vector<Triple> result;
  for (int x = L; x >= 0; --x)
    for (int y = L - x; y >= 0; --y)
      result.push_back({x, y, L - x - y});
  return result;
}
std::vector<Triple> hermites(int L) {
  std::vector<Triple> result;
  for (int degree = 0; degree <= L; ++degree)
    for (int x = 0; x <= degree; ++x)
      for (int y = 0; y <= degree - x; ++y)
        result.push_back({x, y, degree - x - y});
  return result;
}
static void boys_real(unsigned m, Real T, Real *f) {
  if (T == 0) {
    for (unsigned n = 0; n <= m; n++)
      f[n] = Real(1) / (2 * n + 1);
    return;
  }
  if (T < 32) {
    // F_m^(k)(T)=(-1)^k F_{m+k}(T). The nearest half-unit
    // center gives |delta|<=1/4; degree 14 bounds the analytic remainder
    // relatively by exp(1/2)*(1/4)^15/15!, before table/FP rounding.
    unsigned cell = static_cast<unsigned>(2 * T + Real(.5));
    Real delta = T - Real(cell) * Real(.5), v = boys_table[cell][m + 14] * inverse_factorial[14];
    for (int k = 13; k >= 0; k--)
      v = boys_table[cell][m + k] * inverse_factorial[k] - delta * v;
    f[m] = v;
    if (m) {
      Real e = std::exp(-T);
      for (unsigned n = m; n > 0; n--)
        f[n - 1] = (2 * T * f[n] + e) / (2 * n - 1);
    }
    return;
  }
  Real root = std::sqrt(T), e = std::exp(-T);
  f[0] = std::sqrt(pi) * std::erf(root) / (2 * root);
  for (unsigned n = 0; n < m; ++n)
    f[n + 1] = ((2 * n + 1) * f[n] - e) / (2 * T);
}
void boys(unsigned m, double T, double *f) {
  Real values[9];
  boys_real(m, T, values);
  for (unsigned i = 0; i <= m; ++i)
    f[i] = static_cast<double>(values[i]);
}
struct Angular {
  std::array<Triple, 165> h{};
  std::array<std::array<Triple, 6>, 3> cart{};
  constexpr Angular() {
    int i = 0;
    for (int d = 0; d <= 8; ++d)
      for (int x = 0; x <= d; ++x)
        for (int y = 0; y <= d - x; ++y)
          h[i++] = {x, y, d - x - y};
    for (int l = 0; l <= 2; ++l) {
      i = 0;
      for (int x = l; x >= 0; --x)
        for (int y = l - x; y >= 0; --y)
          cart[l][i++] = {x, y, l - x - y};
    }
  }
};
static constexpr Angular angular; // Immutable, no dynamic initialization.

using RStep = HermiteRStep;
struct RPlan {
  std::array<int, 9> offset{};
  std::array<RStep, 495> steps{};
  int count = 0;
  constexpr RPlan(int L) {
    for (int m = 1; m <= L; m++)
      offset[m] = offset[m - 1] + nh(L - m + 1);
    emit_r_dependencies(L, offset, [&](RStep step) { steps[count++] = step; });
  }
};
static constexpr RPlan rplans[9] = {RPlan(0), RPlan(1), RPlan(2), RPlan(3), RPlan(4),
                                    RPlan(5), RPlan(6), RPlan(7), RPlan(8)};
// Constant low-angular dependency graph. Each primitive retains exactly the
// scalar graph's arithmetic order; only index loads/branches are eliminated.
template <int L> static void r_small(Real *R, const Real *X) {
  constexpr auto &plan = rplans[L];
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 128
#endif
  for (int i = 0; i < plan.count; i++) {
    auto v = plan.steps[i];
    Real value = X[v.axis] * R[v.first];
    if (v.coefficient)
      value += v.coefficient * R[v.second];
    R[v.dst] = value;
  }
}

struct MatrixIndices {
  std::array<int, 25> offset{};
  std::array<unsigned short, 4900> index{};
  constexpr MatrixIndices() {
    int pos = 0;
    for (int a = 0; a <= 4; a++)
      for (int b = 0; b <= 4; b++) {
        offset[a * 5 + b] = pos;
        emit_hermite_coupling(
            a, b, [&](int i) { return angular.h[i]; }, [&](unsigned short v) { index[pos++] = v; });
      }
  }
};
static constexpr MatrixIndices matrix_indices;
template <int A, int B> static void m_small(Real *M, const Real *R, Real prefactor) {
  constexpr int count = nh(A) * nh(B), offset = matrix_indices.offset[A * 5 + B];
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 128
#endif
  for (int i = 0; i < count; i++) {
    unsigned code = matrix_indices.index[offset + i];
    Real v = (code & 0x8000) ? -prefactor : prefactor;
    M[i] = v * R[code & 0x7fff];
  }
}

static Real coulomb_r(const libgto_pair &a, const libgto_pair &b, size_t p, size_t q,
                      const Layout &l, Real *R) {
  const auto &x = a.primitives[p];
  const auto &y = b.primitives[q];
  Real rho = x.p / (x.p + y.p) * y.p, X[3], T = 0;
  for (int d = 0; d < 3; ++d) {
    X[d] = (a.a.r[d] - b.a.r[d]) + (x.offset[d] - y.offset[d]);
    T += rho * X[d] * X[d];
  }
  require(std::isfinite(T) && std::isfinite(rho), LIBGTO_NUMERICAL_FAILURE);
  Real F[9];
  boys_real(l.L, T, F);
  const auto &plan = rplans[l.L];
  Real factor = 1;
  for (int m = 0; m <= l.L; m++) {
    R[plan.offset[m]] = factor * F[m];
    factor *= -2 * rho;
  }
  switch (l.L) {
  case 0:
    break;
  case 1:
    r_small<1>(R, X);
    break;
  case 2:
    r_small<2>(R, X);
    break;
  case 3:
    r_small<3>(R, X);
    break;
  case 4:
    r_small<4>(R, X);
    break;
  default:
    for (int i = 0; i < plan.count; i++) {
      auto v = plan.steps[i];
      Real value = X[v.axis] * R[v.first];
      if (v.coefficient)
        value += v.coefficient * R[v.second];
      R[v.dst] = value;
    }
  }
  Real prefactor = 2 * std::pow(pi, 2.5L) / (x.p * y.p * std::sqrt(x.p + y.p)) * x.K * y.K;
  return prefactor;
}
static void coulomb(const libgto_pair &a, const libgto_pair &b, size_t p, size_t q, const Layout &l,
                    Real *R, Real *M) {
  Real prefactor = coulomb_r(a, b, p, q, l, R);
  switch ((a.a.l + a.b.l) * 5 + b.a.l + b.b.l) {
  case 0:
    m_small<0, 0>(M, R, prefactor);
    return;
  case 1:
    m_small<0, 1>(M, R, prefactor);
    return;
  case 2:
    m_small<0, 2>(M, R, prefactor);
    return;
  case 3:
    m_small<0, 3>(M, R, prefactor);
    return;
  case 4:
    m_small<0, 4>(M, R, prefactor);
    return;
  case 5:
    m_small<1, 0>(M, R, prefactor);
    return;
  case 6:
    m_small<1, 1>(M, R, prefactor);
    return;
  case 7:
    m_small<1, 2>(M, R, prefactor);
    return;
  case 8:
    m_small<1, 3>(M, R, prefactor);
    return;
  case 10:
    m_small<2, 0>(M, R, prefactor);
    return;
  case 11:
    m_small<2, 1>(M, R, prefactor);
    return;
  case 12:
    m_small<2, 2>(M, R, prefactor);
    return;
  case 15:
    m_small<3, 0>(M, R, prefactor);
    return;
  case 16:
    m_small<3, 1>(M, R, prefactor);
    return;
  case 20:
    m_small<4, 0>(M, R, prefactor);
    return;
  default:
    break;
  }
  const auto *map =
      matrix_indices.index.data() + matrix_indices.offset[(a.a.l + a.b.l) * 5 + b.a.l + b.b.l];
  for (size_t i = 0; i < l.m; i++) {
    unsigned code = map[i];
    Real signed_prefactor = (code & 0x8000) ? -prefactor : prefactor;
    M[i] = signed_prefactor * R[code & 0x7fff];
  }
}
// Skip only stored exact zeros; no magnitude screening or coefficient changes.
static unsigned first_bit(uint64_t bits) {
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<unsigned>(__builtin_ctzll(bits));
#else
  unsigned n = 0;
  while (!(bits & 1)) {
    bits >>= 1;
    ++n;
  }
  return n;
#endif
}
static void primitive(const libgto_pair &a, const libgto_pair &b, size_t p, size_t q,
                      const Layout &l, Real *R, Real *M, Real *U, Real *G, bool projected) {
  coulomb(a, b, p, q, l, R, M);
  const auto &av = projected && !a.pure_E.empty() ? a.pure_E : a.E;
  const auto &bv = projected && !b.pure_E.empty() ? b.pure_E : b.E;
  const auto &am = projected && !a.pure_E.empty() ? a.pure_mask : a.E_mask;
  const auto &bm = projected && !b.pure_E.empty() ? b.pure_mask : b.E_mask;
  const Real *ea = av.data() + p * l.hab * l.nab, *ec = bv.data() + q * l.hcd * l.ncd;
  if (l.hab == 1 && l.hcd == 1) {
    G[0] = M[0];
    return;
  }
  if (l.hab == 1) {
    std::fill_n(G, l.g, 0);
    for (int k = 0; k < l.hcd; k++) {
      Real m = M[k];
      if (m == 0)
        continue;
      for (uint64_t bits = bm[q * l.hcd + k]; bits; bits &= bits - 1) {
        unsigned c = first_bit(bits);
        G[c] += m * ec[k * l.ncd + c];
      }
    }
    return;
  }
  if (l.hcd == 1) {
    std::fill_n(G, l.g, 0);
    for (int h = 0; h < l.hab; h++) {
      Real m = M[h];
      if (m == 0)
        continue;
      for (uint64_t bits = am[p * l.hab + h]; bits; bits &= bits - 1) {
        unsigned a = first_bit(bits);
        G[a] += m * ea[h * l.nab + a];
      }
    }
    return;
  }
  std::fill_n(U, l.u, 0);
  std::fill_n(G, l.g, 0);
  const uint64_t full_ab = (uint64_t(1) << l.nab) - 1, full_cd = (uint64_t(1) << l.ncd) - 1;
  for (int h = 0; h < l.hab; h++)
    for (int k = 0; k < l.hcd; k++) {
      uint64_t bits = bm[q * l.hcd + k];
      Real m = M[h * l.hcd + k];
      if (bits == full_cd) {
        for (int cd = 0; cd < l.ncd; cd++)
          U[h * l.ncd + cd] += m * ec[k * l.ncd + cd];
      } else
        for (; bits; bits &= bits - 1) {
          unsigned cd = first_bit(bits);
          U[h * l.ncd + cd] += m * ec[k * l.ncd + cd];
        }
    }
  for (int h = 0; h < l.hab; h++) {
    uint64_t mask = am[p * l.hab + h];
    if (!mask)
      continue;
    for (int cd = 0; cd < l.ncd; cd++) {
      Real u = U[h * l.ncd + cd];
      if (mask == full_ab) {
        for (int ab = 0; ab < l.nab; ab++)
          G[ab + l.nab * cd] += ea[h * l.nab + ab] * u;
      } else
        for (uint64_t bits = mask; bits; bits &= bits - 1) {
          unsigned ab = first_bit(bits);
          G[ab + l.nab * cd] += ea[h * l.nab + ab] * u;
        }
    }
  }
}
// Internal disjoint-array contract: U/G, packed M and immutable pair E
// occupy distinct allocations or disjoint planned workspace slices.
#if defined(__GNUC__) || defined(__clang__)
#define GTO_RESTRICT __restrict__
#else
#define GTO_RESTRICT
#endif
template <int H, int K, int A, int C>
static void dense_fixed(double *GTO_RESTRICT U, double *GTO_RESTRICT g,
                        const double *GTO_RESTRICT M, const double *GTO_RESTRICT ea,
                        const double *GTO_RESTRICT ec) {
  for (int h = 0; h < H; h++)
    for (int k = 0; k < K; k++) {
      double m = M[h * K + k];
      for (int c = 0; c < C; c++)
        U[h * C + c] += m * ec[k * C + c];
    }
  for (int h = 0; h < H; h++)
    for (int c = 0; c < C; c++) {
      double u = U[h * C + c];
      for (int a = 0; a < A; a++)
        g[a + A * c] += ea[h * A + a] * u;
    }
}
#undef GTO_RESTRICT
template <int H, int K, int A, int C>
static bool primitive_r4(const libgto_pair &a, const libgto_pair &b, size_t p, size_t q,
                         const Layout &l, double *G, double *E, bool projected, double budget) {
  constexpr int L = 4;
  constexpr double eps = std::numeric_limits<double>::epsilon();
  (void)l;
  const auto &x = a.primitives[p];
  const auto &y = b.primitives[q];
  double ap = double(x.p), bp = double(y.p), sum = ap + bp;
  if (!std::isnormal(ap) || !std::isnormal(bp) || !std::isnormal(sum))
    return false;
  double rho = ap / sum * bp, X[3], T = 0;
  for (int d = 0; d < 3; d++) {
    X[d] = double((a.a.r[d] - b.a.r[d]) + (x.offset[d] - y.offset[d]));
    T += rho * X[d] * X[d];
  }
  if (!std::isnormal(rho) || !std::isfinite(T) || T < 0)
    return false;
  if (!std::isnormal(double(x.K)) || !std::isnormal(double(y.K)))
    return false;
  double prefactor =
      double(2 * std::pow(pi, 2.5L)) / (ap * bp * std::sqrt(sum)) * double(x.K) * double(y.K);
  if (!std::isnormal(prefactor))
    return false;
  double F[5];
  bool foundation_fast = 1 && quartic_boys::evaluate(T, F);
  if (!foundation_fast) {
    Real wide_boys[5];
    boys_real(4, Real(T), wide_boys);
    for (int m = 0; m < 5; m++)
      F[m] = static_cast<double>(wide_boys[m]);
  }
  const auto &plan = rplans[L];
  double R[70], error[70], factor = 1;
  // T*F_(m+1)/F_m <= min(T,m+1/2) bounds first-order geometry
  // sensitivity. The 128-epsilon envelope also covers table/recurrence rounding.
  // This complete monitor remains empirical; it is not an error certificate.
  for (int m = 0; m <= L; m++) {
    int i = plan.offset[m];
    R[i] = factor * F[m];
    error[i] = std::abs(R[i]) * eps *
               (128 * (1 + std::min(T, m + 0.5)) + 16 * m + (foundation_fast ? 4 : 0));
    factor *= -2 * rho;
  }
  for (int i = 0; i < plan.count; i++) {
    auto s = plan.steps[i];
    double v = X[s.axis] * R[s.first],
           e = std::abs(X[s.axis]) * error[s.first] + 3 * eps * std::abs(v);
    if (s.coefficient) {
      double u = s.coefficient * R[s.second];
      v += u;
      e += s.coefficient * error[s.second] + 2 * eps * (std::abs(u) + std::abs(v));
    }
    R[s.dst] = v;
    error[s.dst] = e;
  }
  const auto *map = matrix_indices.index.data() + matrix_indices.offset[(H == 1    ? 0
                                                                         : H == 4  ? 1
                                                                         : H == 10 ? 2
                                                                         : H == 20 ? 3
                                                                                   : 4) *
                                                                            5 +
                                                                        (K == 1    ? 0
                                                                         : K == 4  ? 1
                                                                         : K == 10 ? 2
                                                                         : K == 20 ? 3
                                                                                   : 4)];
  double M[H * K], maxm = 0, maxerror = 0;
  // Unique final-spatial scaling, as in the higher-angular FP64 path.
  for (int i = 0; i < nh(L); i++) {
    double value = prefactor * R[i];
    if (value != 0 && !std::isnormal(value))
      return false;
    R[i] = value;
    maxm = std::max(maxm, std::abs(value));
    maxerror = std::max(maxerror, std::abs(prefactor) * error[i] + 48 * eps * std::abs(value));
  }
  for (int i = 0; i < H * K; i++) {
    unsigned code = map[i];
    double value = R[code & 0x7fff];
    M[i] = (code & 0x8000) ? -value : value;
  }
  const auto &av = projected && !a.pure_E64.empty() ? a.pure_E64 : a.E64;
  const auto &bv = projected && !b.pure_E64.empty() ? b.pure_E64 : b.E64;
  const auto &an = projected && !a.norm_pure.empty() ? a.norm_pure : a.norm_cart;
  const auto &bn = projected && !b.norm_pure.empty() ? b.norm_pure : b.norm_cart;
  double na = an[p * (A + 1)], nb = bn[q * (C + 1)];
  double bound = (maxerror + (H + K + 16) * eps * maxm) * na * nb;
  bound = bound * (1 + 32 * eps) + std::numeric_limits<double>::min();
  if (!std::isfinite(bound) || bound > budget)
    return false;
  const double *ea = av.data() + p * H * A, *ec = bv.data() + q * K * C;
  double g[A * C];
  // Exact low-angular E support, independent of geometry and contraction weights.
  // Internal p columns are x,y,z; first-degree Hermites are z,y,x. Public pure
  // p reordering remains in the output module. No threshold screening occurs.
  if (H == 1 && K == 1)
    g[0] = M[0];
  else if (H == 1 && K == 4) {
    for (int c = 0; c < 3; c++)
      g[c] = M[0] * ec[c] + M[3 - c] * ec[(3 - c) * 3 + c];
  } else if (K == 1 && H == 4) {
    for (int i = 0; i < 3; i++)
      g[i] = M[0] * ea[i] + M[3 - i] * ea[(3 - i) * 3 + i];
  } else if (H == 4 && K == 4) {
    double U[12];
    for (int h = 0; h < 4; h++)
      for (int c = 0; c < 3; c++)
        U[h * 3 + c] = M[h * 4] * ec[c] + M[h * 4 + 3 - c] * ec[(3 - c) * 3 + c];
    for (int c = 0; c < 3; c++)
      for (int i = 0; i < 3; i++)
        g[i + 3 * c] = ea[i] * U[c] + ea[(3 - i) * 3 + i] * U[(3 - i) * 3 + c];
  } else if (H == 1) {
    std::fill_n(g, (A * C), 0.);
    for (int k = 0; k < K; k++)
      for (int c = 0; c < C; c++)
        g[c] += M[k] * ec[k * C + c];
  } else if (K == 1) {
    std::fill_n(g, (A * C), 0.);
    for (int h = 0; h < H; h++)
      for (int i = 0; i < A; i++)
        g[i] += M[h] * ea[h * A + i];
  } else {
    double U[H * C]{};
    std::fill_n(g, A * C, 0.);
    if constexpr (K == 4) {
      for (int h = 0; h < H; h++)
        for (int c = 0; c < C; c++)
          U[h * C + c] = M[h * K] * ec[c] + M[h * K + 3 - c] * ec[(3 - c) * 3 + c];
    } else {
      for (int h = 0; h < H; h++)
        for (int k = 0; k < K; k++) {
          double m = M[h * K + k];
          for (int c = 0; c < C; c++)
            U[h * C + c] += m * ec[k * C + c];
        }
    }
    if constexpr (H == 4) {
      for (int c = 0; c < C; c++)
        for (int i = 0; i < A; i++)
          g[i + A * c] = ea[i] * U[c] + ea[(3 - i) * 3 + i] * U[(3 - i) * C + c];
    } else {
      for (int h = 0; h < H; h++)
        for (int c = 0; c < C; c++) {
          double u = U[h * C + c];
          for (int i = 0; i < A; i++)
            g[i + A * c] += ea[h * A + i] * u;
        }
    }
  }
  for (size_t i = 0; i < (A * C); i++) {
    if (!std::isfinite(g[i]))
      return false;
    G[i] = g[i];
    E[i] = bound;
  }
  return true;
}

static bool primitive_dd_moment(const libgto_pair &a, const libgto_pair &b, size_t p, size_t q,
                                const Layout &l, double *G, double *E, double budget) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  const auto &x = a.primitives[p];
  const auto &y = b.primitives[q];
  double ap = x.p64, bp = y.p64, sum = ap + bp;
  if (!x.low_normal || !y.low_normal || !std::isnormal(sum) || ap < 1e-40 || bp < 1e-40 ||
      ap > 1e40 || bp > 1e40)
    return false;
  double rho = ap / sum * bp, X[3], T = 0, ma[3], mc[3], va[3], vc[3], amax = 0, cmax = 0;
  const int diag[3] = {0, 3, 5};
  for (int d = 0; d < 3; d++) {
    X[d] = double((a.a.r[d] - b.a.r[d]) + (x.offset[d] - y.offset[d]));
    T += rho * X[d] * X[d];
    // Cartesian E for a diagonal d polynomial: E_(e_d)=mean_d/p.
    ma[d] = ap * a.E64[(p * 10 + 3 - d) * 6 + diag[d]];
    mc[d] = bp * b.E64[(q * 10 + 3 - d) * 6 + diag[d]];
    va[d] = -(rho / ap) * X[d];
    vc[d] = (rho / bp) * X[d];
    double u = std::abs(ma[d]) + std::abs(va[d]), v = std::abs(mc[d]) + std::abs(vc[d]);
    if (!std::isfinite(u) || !std::isfinite(v) || u > 1e50 || v > 1e50)
      return false;
    amax = std::max(amax, u);
    cmax = std::max(cmax, v);
  }
  if (!std::isnormal(rho) || !std::isfinite(T) || T < 0)
    return false;
  double prefactor = double(2 * std::pow(pi, 2.5L)) / (ap * bp * std::sqrt(sum)) * x.K64 * y.K64;
  if (!std::isnormal(prefactor))
    return false;
  double F[5];
  bool fast = 1 && quartic_boys::evaluate(T, F);
  if (!fast) {
    Real f[5];
    boys_real(4, Real(T), f);
    for (int m = 0; m < 5; m++)
      F[m] = double(f[m]);
  }
  double cov = rho / (2 * ap * bp),
         env = (4 * amax * amax * cmax * cmax + 12 * std::abs(cov) * amax * cmax) + 3 * cov * cov;
  double bound =
      std::abs(prefactor) * F[0] * env * ((2048 + (fast ? 256 : 0)) * eps * (1 + std::min(T, 4.5)));
  bound = bound * (1 + 64 * eps) + std::numeric_limits<double>::min();
  if (!std::isfinite(bound) || bound > budget)
    return false;
  double qa[5][3], qc[5][3], aa[5][3], av[5][3], ca[5][3], cv[5][3], U[3][5];
  quadratic_pair::forms(ma, va, qa, aa, av);
  quadratic_pair::forms(mc, vc, qc, ca, cv);
  for (int j = 0; j < 5; j++)
    for (int k = 0; k < 3; k++)
      U[k][j] = (qc[j][0] * F[k] + qc[j][1] * F[k + 1]) + qc[j][2] * F[k + 2];
  for (int j = 0; j < 5; j++)
    for (int i = 0; i < 5; i++) {
      double w0 = 0, w1 = 0, w2 = 0;
      for (int d = 0; d < 3; d++) {
        w0 += aa[i][d] * ca[j][d];
        w1 += aa[i][d] * cv[j][d] + av[i][d] * ca[j][d];
        w2 += av[i][d] * cv[j][d];
      }
      double value = (qa[i][0] * U[0][j] + qa[i][1] * U[1][j]) + qa[i][2] * U[2][j];
      value += 4 * cov * ((w0 * F[1] + w1 * F[2]) + w2 * F[3]);
      if (i == j)
        value += 3 * cov * cov * F[2];
      value *= prefactor;
      if (!std::isfinite(value) || (value != 0 && !std::isnormal(value)))
        return false;
      G[i + 5 * j] = value;
      E[i + 5 * j] = bound;
    }
  return true;
}

// FP64 dense contractions. E is rounded from wide prepared data, and M from
// wide Boys/R. The propagated bound monitors these casts and dot-product
// roundoff. It is an error indicator, not a proof about wide input error.
static bool primitive_fp64(const libgto_pair &a, const libgto_pair &b, size_t p, size_t q,
                           const Layout &l, Real *R, Real *M, double *scratch, bool projected,
                           double budget, bool ordered_angular = false,
                           double *uniform_error = nullptr, bool defer_clear = false,
                           bool ps_bra = false) {
  if (uniform_error)
    *uniform_error = std::numeric_limits<double>::quiet_NaN();
  // Isolate the L4 auxiliary precision change; keep the parent's monitored
  // FP64 channel accumulators. A rejected R4 attempt uses the old wide-R path.
  if (l.L == 4) {
    double *g = scratch + 2 * l.u, *ge = g + l.g;
    bool success = false;
    if (projected && l.hab == 10 && l.hcd == 10 && l.nab == 5 && l.ncd == 5 && 2 == 2 &&
        primitive_dd_moment(a, b, p, q, l, g, ge, budget))
      return true;
    if (l.hab == 10 && l.hcd == 10 && l.nab == 5 && l.ncd == 5)
      success = primitive_r4<10, 10, 5, 5>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 10 && l.hcd == 10 && l.nab == 5 && l.ncd == 6)
      success = primitive_r4<10, 10, 5, 6>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 10 && l.hcd == 10 && l.nab == 5 && l.ncd == 9)
      success = primitive_r4<10, 10, 5, 9>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 10 && l.hcd == 10 && l.nab == 6 && l.ncd == 5)
      success = primitive_r4<10, 10, 6, 5>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 10 && l.hcd == 10 && l.nab == 6 && l.ncd == 6)
      success = primitive_r4<10, 10, 6, 6>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 10 && l.hcd == 10 && l.nab == 6 && l.ncd == 9)
      success = primitive_r4<10, 10, 6, 9>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 10 && l.hcd == 10 && l.nab == 9 && l.ncd == 5)
      success = primitive_r4<10, 10, 9, 5>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 10 && l.hcd == 10 && l.nab == 9 && l.ncd == 6)
      success = primitive_r4<10, 10, 9, 6>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 10 && l.hcd == 10 && l.nab == 9 && l.ncd == 9)
      success = primitive_r4<10, 10, 9, 9>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 4 && l.hcd == 20 && l.nab == 3 && l.ncd == 15)
      success = primitive_r4<4, 20, 3, 15>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 4 && l.hcd == 20 && l.nab == 3 && l.ncd == 18)
      success = primitive_r4<4, 20, 3, 18>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 20 && l.hcd == 4 && l.nab == 15 && l.ncd == 3)
      success = primitive_r4<20, 4, 15, 3>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 20 && l.hcd == 4 && l.nab == 18 && l.ncd == 3)
      success = primitive_r4<20, 4, 18, 3>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 1 && l.hcd == 35 && l.nab == 1 && l.ncd == 25)
      success = primitive_r4<1, 35, 1, 25>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 1 && l.hcd == 35 && l.nab == 1 && l.ncd == 36)
      success = primitive_r4<1, 35, 1, 36>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 35 && l.hcd == 1 && l.nab == 25 && l.ncd == 1)
      success = primitive_r4<35, 1, 25, 1>(a, b, p, q, l, g, ge, projected, budget);
    if (l.hab == 35 && l.hcd == 1 && l.nab == 36 && l.ncd == 1)
      success = primitive_r4<35, 1, 36, 1>(a, b, p, q, l, g, ge, projected, budget);
    if (success)
      return true;
  }
  // Unique signed-R conversion pays only when it reduces the cast count.
  bool reuse = l.m > 2 * size_t(nh(l.L));
  Real prefactor = 0;
  if (reuse)
    prefactor = coulomb_r(a, b, p, q, l, R);
  else
    coulomb(a, b, p, q, l, R, M);
  double *U = scratch, *ue = U + l.u, *g = ue + l.u, *ge = g + l.g, *ga = ge + l.g;
  const double *ea =
      (projected && !a.pure_E64.empty() ? a.pure_E64 : a.E64).data() + p * l.hab * l.nab;
  const double *ec =
      (projected && !b.pure_E64.empty() ? b.pure_E64 : b.E64).data() + q * l.hcd * l.ncd;
  if (!defer_clear) {
    std::fill_n(U, l.u, 0.);
    std::fill_n(g, l.g, 0.);
  }
  constexpr double eps = std::numeric_limits<double>::epsilon();
  double *packed = scratch + l.fast - l.m;
  double maximum = 0;
  const double *unique_values = nullptr;
  const unsigned short *coupling_codes = nullptr;
  if (reuse) {
    // Every spatial derivative up to L occurs in h+k; scale/cast it once.
    // Two signs preserve the original sign-before-conversion operation as well.
    int nr = nh(l.L);
    double *unique = packed - 2 * nr;
    for (int i = 0; i < nr; i++) {
      Real wide = prefactor * R[i];
      if (std::abs(wide) > std::numeric_limits<double>::max())
        return false;
      double positive = static_cast<double>(wide), negative = static_cast<double>(-wide);
      if (wide != 0 && (!std::isnormal(positive) || !std::isnormal(negative)))
        return false;
      unique[i] = positive;
      unique[nr + i] = negative;
      maximum = std::max(maximum, std::max(std::abs(positive), std::abs(negative)));
    }
    const auto *indices =
        matrix_indices.index.data() + matrix_indices.offset[(a.a.l + a.b.l) * 5 + b.a.l + b.b.l];
    unique_values = unique;
    coupling_codes = indices;
    if (!ps_bra)
      for (size_t i = 0; i < l.m; i++) {
        unsigned code = indices[i];
        packed[i] = unique[(code & 0x7fff) + ((code & 0x8000) ? nr : 0)];
      }
  } else {
    for (size_t i = 0; i < l.m; i++) {
      Real wide = M[i];
      if (std::abs(wide) > std::numeric_limits<double>::max())
        return false;
      double m = static_cast<double>(wide);
      if (wide != 0 && !std::isnormal(m))
        return false;
      packed[i] = m;
      maximum = std::max(maximum, std::abs(m));
    }
  }
  const auto &an = projected && !a.norm_pure.empty() ? a.norm_pure : a.norm_cart;
  const auto &cn = projected && !b.norm_pure.empty() ? b.norm_pure : b.norm_cart;
  const double *na = an.data() + p * (l.nab + 1) + 1, *nc = cn.data() + q * (l.ncd + 1) + 1;
  const double floor = std::numeric_limits<double>::min();
  const double gamma = (l.hab + l.hcd + 16) * eps / (1 - (l.hab + l.hcd + 16) * eps);
  const double factor = std::max(floor, gamma * maximum);
  auto bound = [&](double aa, double cc) {
    double z = std::max(floor, factor * aa);
    z = std::max(floor, z * cc);
    return z * (1 + 16 * eps) + floor * ((1 + aa) * (l.hab + l.hcd + 2));
  };
  // A separable norm majorant can replace the embedded monitor when its
  // weighted absolute budget is small. Otherwise retain the original monitor.
  double uniform_majorant = bound(na[-1], nc[-1]);
  if (uniform_majorant <= budget) {
    if (ps_bra) {
      if (l.hcd == 4)
        angular_dd_ps_ket_unique(coupling_codes, unique_values, ea, ec, U, g);
      else if (l.hab == 4)
        angular_ps_dd_bra_unique(coupling_codes, unique_values, ea, ec, packed, g);
      else {
        const auto &am = projected && !a.pure_E.empty() ? a.pure_mask : a.E_mask;
        angular_ds_bra_unique(l.hcd, coupling_codes, unique_values, ea, ec, am.data() + p * 10,
                              packed, g);
      }
    } else if (ordered_angular) {
      angular_products_ordered(l.hab, l.hcd, l.nab, l.ncd, packed, ea, ec, U, g);
    } else if (l.L == 5) {
      // Exact prepared support, no magnitude threshold. Ascending bit iteration
      // preserves each accumulator's nonzero term order; mask storage pre-exists.
      const auto &am = projected && !a.pure_E.empty() ? a.pure_mask : a.E_mask;
      const auto &cm = projected && !b.pure_E.empty() ? b.pure_mask : b.E_mask;
      for (int h = 0; h < l.hab; h++)
        if (am[p * l.hab + h])
          for (int k = 0; k < l.hcd; k++) {
            uint64_t bits = cm[q * l.hcd + k];
            double m = packed[h * l.hcd + k];
            while (bits) {
              unsigned c = first_set_bit(bits);
              bits &= bits - 1;
              U[h * l.ncd + c] += m * ec[k * l.ncd + c];
            }
          }
      for (int h = 0; h < l.hab; h++)
        if (am[p * l.hab + h])
          for (int c = 0; c < l.ncd; c++) {
            uint64_t bits = am[p * l.hab + h];
            double u = U[h * l.ncd + c];
            while (bits) {
              unsigned ab = first_set_bit(bits);
              bits &= bits - 1;
              g[ab + l.nab * c] += ea[h * l.nab + ab] * u;
            }
          }
    } else if (l.hab == 1) {
      for (int k = 0; k < l.hcd; k++) {
        double m = packed[k];
        for (int c = 0; c < l.ncd; c++)
          g[c] += m * ec[k * l.ncd + c];
      }
    } else if (l.hcd == 1) {
      for (int h = 0; h < l.hab; h++) {
        double m = packed[h];
        for (int a = 0; a < l.nab; a++)
          g[a] += m * ea[h * l.nab + a];
      }
    } else if ((projected && !a.pure_E.empty() ? a.sparse_pure : a.sparse_cart) ||
               (projected && !b.pure_E.empty() ? b.sparse_pure : b.sparse_cart)) {
      const auto &am = projected && !a.pure_E.empty() ? a.pure_mask : a.E_mask;
      const auto &cm = projected && !b.pure_E.empty() ? b.pure_mask : b.E_mask;
      for (int h = 0; h < l.hab; h++)
        if (am[p * l.hab + h])
          for (int k = 0; k < l.hcd; k++) {
            if (!cm[q * l.hcd + k])
              continue;
            double m = packed[h * l.hcd + k];
            for (int c = 0; c < l.ncd; c++)
              U[h * l.ncd + c] += m * ec[k * l.ncd + c];
          }
      for (int h = 0; h < l.hab; h++)
        if (am[p * l.hab + h])
          for (int c = 0; c < l.ncd; c++) {
            double u = U[h * l.ncd + c];
            for (int a = 0; a < l.nab; a++)
              g[a + l.nab * c] += ea[h * l.nab + a] * u;
          }
    } else {
      if (l.hab == 35 && l.hcd == 35 && (l.nab == 25 || l.nab == 36) &&
          (l.ncd == 25 || l.ncd == 36))
        angular_products(l.hab, l.hcd, l.nab, l.ncd, packed, ea, ec, U, g);
      else if (l.hab == 35 && l.hcd == 35 && l.nab == 36 && l.ncd == 36)
        dense_fixed<35, 35, 36, 36>(U, g, packed, ea, ec);
      else if (l.hab == 10 && l.hcd == 10 && l.nab == 9 && l.ncd == 9)
        dense_fixed<10, 10, 9, 9>(U, g, packed, ea, ec);
      else {
        for (int h = 0; h < l.hab; h++)
          for (int k = 0; k < l.hcd; k++) {
            double m = packed[h * l.hcd + k];
            for (int cd = 0; cd < l.ncd; cd++)
              U[h * l.ncd + cd] += m * ec[k * l.ncd + cd];
          }
        for (int h = 0; h < l.hab; h++)
          for (int cd = 0; cd < l.ncd; cd++) {
            double u = U[h * l.ncd + cd];
            for (int ab = 0; ab < l.nab; ab++)
              g[ab + l.nab * cd] += ea[h * l.nab + ab] * u;
          }
      }
    }
    if (uniform_error) {
      for (size_t i = 0; i < l.g; i++)
        if (!std::isfinite(g[i]))
          return false;
      *uniform_error = uniform_majorant;
    } else
      for (int cd = 0; cd < l.ncd; cd++)
        for (int ab = 0; ab < l.nab; ab++) {
          size_t i = ab + l.nab * cd;
          ge[i] = bound(na[ab], nc[cd]);
          if (!std::isfinite(g[i]) || !std::isfinite(ge[i]))
            return false;
        }
    return true;
  }
  if (ps_bra) {
    const int nr = nh(l.L);
    for (size_t i = 0; i < l.m; i++) {
      unsigned code = coupling_codes[i];
      packed[i] = unique_values[(code & 0x7fff) + ((code & 0x8000) ? nr : 0)];
    }
  }
  if (defer_clear) {
    std::fill_n(U, l.u, 0.);
    std::fill_n(g, l.g, 0.);
  }
  if (l.hab == 35 && l.hcd == 35 && l.nab >= 25 && l.ncd >= 25) {
    // Numerically equivalent absolute-product majorant, evaluated separately by
    // the same small matrix adapter. Required packing is part of this call.
    double *absM = scratch + 2 * l.u + 3 * l.g + 2 * (l.tmp + l.contracted);
    double *absEa = absM + l.m, *absEc = absEa + l.hab * l.nab;
    for (size_t i = 0; i < l.m; i++)
      absM[i] = std::abs(packed[i]);
    for (int i = 0; i < l.hab * l.nab; i++)
      absEa[i] = std::abs(ea[i]);
    for (int i = 0; i < l.hcd * l.ncd; i++)
      absEc[i] = std::abs(ec[i]);
    angular_products(l.hab, l.hcd, l.nab, l.ncd, packed, ea, ec, U, g);
    angular_products(l.hab, l.hcd, l.nab, l.ncd, absM, absEa, absEc, ue, ga);
    const double gamma = (l.hab + l.hcd + 16) * eps / (1 - (l.hab + l.hcd + 16) * eps);
    for (int c = 0; c < l.ncd; c++)
      for (int a = 0; a < l.nab; a++) {
        size_t i = a + l.nab * c;
        ge[i] =
            ga[i] * gamma * (1 + gamma) + floor * (l.hab + l.hcd + 2) * (1 + na[a]) * (1 + nc[c]);
        if (!std::isfinite(g[i]) || !std::isfinite(ge[i]))
          return false;
      }
    return true;
  }
  std::fill_n(ue, l.u, 0.);
  std::fill_n(ge, l.g, 0.);
  std::fill_n(ga, l.g, 0.);
  for (int h = 0; h < l.hab; h++) {
    for (int k = 0; k < l.hcd; k++) {
      double m = packed[h * l.hcd + k];
      for (int cd = 0; cd < l.ncd; cd++) {
        double value = m * ec[k * l.ncd + cd];
        U[h * l.ncd + cd] += value;
        ue[h * l.ncd + cd] += std::abs(value);
      }
    }
    for (int cd = 0; cd < l.ncd; cd++)
      ue[h * l.ncd + cd] = ue[h * l.ncd + cd] * ((l.hcd + 4) * eps / (1 - (l.hcd + 4) * eps)) +
                           l.hcd * std::numeric_limits<double>::denorm_min();
  }
  for (int h = 0; h < l.hab; h++)
    for (int cd = 0; cd < l.ncd; cd++) {
      double u = U[h * l.ncd + cd], error = ue[h * l.ncd + cd];
      for (int ab = 0; ab < l.nab; ab++) {
        double e = ea[h * l.nab + ab], value = e * u;
        size_t i = ab + l.nab * cd;
        g[i] += value;
        ga[i] += std::abs(value);
        ge[i] += std::abs(e) * error;
      }
    }
  for (size_t i = 0; i < l.g; i++) {
    ge[i] = ge[i] * (1 + eps) + ga[i] * ((l.hab + 3) * eps / (1 - (l.hab + 3) * eps)) +
            l.hab * std::numeric_limits<double>::denorm_min();
    if (!std::isfinite(g[i]) || !std::isfinite(ge[i]))
      return false; /* The channel path consumes the FP64 block directly. */
  }
  return true;
}

// wide precision before rounding. Geometry, Boys, recurrence and angular casts
// enter an absolute forward-error majorant; wide contraction is unchanged.
// This finite-domain monitor is an empirical guard, not an accuracy certificate.
struct LowBoysTable {
  std::array<std::array<double, 18>, 65> f{};
  constexpr LowBoysTable() {
    for (int c = 0; c < 65; c++)
      for (int m = 0; m < 18; m++)
        f[c][m] = double(boys_table[c][m]);
  }
};
static constexpr LowBoysTable low_table;

template <unsigned order> static void low_boys_fixed(double T, double *F) {
  if constexpr (order == 3) {
    Real wide[4];
    boys_real(3, Real(T), wide);
    for (int m = 0; m < 4; m++)
      F[m] = static_cast<double>(wide[m]);
    return;
  }
  if (T == 0) {
    for (int m = 0; m <= order; m++)
      F[m] = 1. / (2 * m + 1);
  } else if (T < 32) {
    unsigned cell = unsigned(2 * T + .5);
    double delta = T - .5 * cell;
    double v = low_table.f[cell][order + 14] * double(inverse_factorial[14]);
    for (int k = 13; k >= 0; k--)
      v = low_table.f[cell][order + k] * double(inverse_factorial[k]) - delta * v;
    F[order] = v;
    if (order) {
      double e = std::exp(-T);
      for (int m = order; m > 0; m--)
        F[m - 1] = (2 * T * F[m] + e) / (2 * m - 1);
    }
  } else {
    double root = std::sqrt(T);
    F[0] = double(std::sqrt(pi)) / (2 * root);
    if (T >= (order <= 2 ? 48. : 64.)) {
      for (unsigned n = 0; n < order; n++)
        F[n + 1] = (2 * n + 1) * F[n] / (2 * T);
    } else {
      F[0] *= std::erf(root);
      if (order) {
        double e = std::exp(-T);
        for (unsigned n = 0; n < order; n++)
          F[n + 1] = ((2 * n + 1) * F[n] - e) / (2 * T);
      }
    }
  }
}
// Borrowed, call-scoped view. Geometry and representation are resolved once
// per shell quartet; primitive-dependent data and R stay in their joint loops.
struct LowContext {
  const Primitive *pa;
  const Primitive *pb;
  const double *ea;
  const double *ec;
  const double *an;
  const double *bn;
  Real delta[3];
};
static LowContext low_context(const libgto_pair &a, const libgto_pair &b, bool projected) {
  LowContext c;
  c.pa = a.primitives.data();
  c.pb = b.primitives.data();
  c.ea = (projected && !a.pure_E64.empty() ? a.pure_E64 : a.E64).data();
  c.ec = (projected && !b.pure_E64.empty() ? b.pure_E64 : b.E64).data();
  c.an = (projected && !a.norm_pure.empty() ? a.norm_pure : a.norm_cart).data();
  c.bn = (projected && !b.norm_pure.empty() ? b.norm_pure : b.norm_cart).data();
  for (int d = 0; d < 3; d++)
    c.delta[d] = a.a.r[d] - b.a.r[d];
  return c;
}
template <int H, int K, int A, int C, int L, class Output = Real>
static bool primitive_low_fixed(const LowContext &ctx, size_t p, size_t q, Output *G,
                                double budget) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  const auto &x = ctx.pa[p];
  const auto &y = ctx.pb[q];
  if (!x.low_normal || !y.low_normal)
    return false;
  double ap = x.p64, bp = y.p64, sum = ap + bp;
  if (!std::isnormal(sum))
    return false;
  double rho = ap / sum * bp, X[3], T = 0;
  for (int d = 0; d < 3; d++) {
    X[d] = double(ctx.delta[d] + (x.offset[d] - y.offset[d]));
    T += rho * X[d] * X[d];
  }
  if (!std::isnormal(rho) || !std::isfinite(T) || T < 0)
    return false;
  double prefactor = double(2 * std::pow(pi, 2.5L)) / (ap * bp * std::sqrt(sum)) * x.K64 * y.K64;
  if (!std::isnormal(prefactor))
    return false;
  double F[4];
  low_boys_fixed<L>(T, F);
  const auto &plan = rplans[L];
  double R[35], error[35], factor = 1;
  // T*F_(m+1)/F_m <= min(T,m+1/2) bounds first-order geometry
  // sensitivity. The 128-epsilon envelope also covers table/recurrence rounding.
  // This complete monitor remains empirical; it is not an error certificate.
  for (int m = 0; m <= L; m++) {
    int i = plan.offset[m];
    R[i] = factor * F[m];
    error[i] = std::abs(R[i]) * eps * (128 * (1 + std::min(T, m + 0.5)) + 16 * m);
    factor *= -2 * rho;
  }
  for (int i = 0; i < plan.count; i++) {
    auto s = plan.steps[i];
    double v = X[s.axis] * R[s.first],
           e = std::abs(X[s.axis]) * error[s.first] + 3 * eps * std::abs(v);
    if (s.coefficient) {
      double u = s.coefficient * R[s.second];
      v += u;
      e += s.coefficient * error[s.second] + 2 * eps * (std::abs(u) + std::abs(v));
    }
    R[s.dst] = v;
    error[s.dst] = e;
  }
  const auto *map = matrix_indices.index.data() + matrix_indices.offset[(H == 1    ? 0
                                                                         : H == 4  ? 1
                                                                         : H == 10 ? 2
                                                                                   : 3) *
                                                                            5 +
                                                                        (K == 1    ? 0
                                                                         : K == 4  ? 1
                                                                         : K == 10 ? 2
                                                                                   : 3)];
  double M[40], maxm = 0, maxerror = 0;
  for (size_t i = 0; i < (H * K); i++) {
    unsigned code = map[i];
    double value = prefactor * R[code & 0x7fff];
    if (value != 0 && !std::isnormal(value))
      return false;
    M[i] = (code & 0x8000) ? -value : value;
    maxm = std::max(maxm, std::abs(value));
    maxerror =
        std::max(maxerror, std::abs(prefactor) * error[code & 0x7fff] + 48 * eps * std::abs(value));
  }
  double na = ctx.an[p * (A + 1)], nb = ctx.bn[q * (C + 1)];
  double bound = (maxerror + (H + K + 16) * eps * maxm) * na * nb;
  bound = bound * (1 + 32 * eps) + std::numeric_limits<double>::min();
  if (!std::isfinite(bound) || bound > budget)
    return false;
  const double *ea = ctx.ea + p * H * A, *ec = ctx.ec + q * K * C;
  double g[32];
  // Exact low-angular E support, independent of geometry and contraction weights.
  // Internal p columns are x,y,z; first-degree Hermites are z,y,x. Public pure
  // p reordering remains in the output module. No threshold screening occurs.
  if (H == 1 && K == 1)
    g[0] = M[0];
  else if (H == 1 && K == 4) {
    for (int c = 0; c < 3; c++)
      g[c] = M[0] * ec[c] + M[3 - c] * ec[(3 - c) * 3 + c];
  } else if (K == 1 && H == 4) {
    for (int i = 0; i < 3; i++)
      g[i] = M[0] * ea[i] + M[3 - i] * ea[(3 - i) * 3 + i];
  } else if (H == 4 && K == 4) {
    double U[12];
    for (int h = 0; h < 4; h++)
      for (int c = 0; c < 3; c++)
        U[h * 3 + c] = M[h * 4] * ec[c] + M[h * 4 + 3 - c] * ec[(3 - c) * 3 + c];
    for (int c = 0; c < 3; c++)
      for (int i = 0; i < 3; i++)
        g[i + 3 * c] = ea[i] * U[c] + ea[(3 - i) * 3 + i] * U[(3 - i) * 3 + c];
  } else if (H == 1) {
    std::fill_n(g, (A * C), 0.);
    for (int k = 0; k < K; k++)
      for (int c = 0; c < C; c++)
        g[c] += M[k] * ec[k * C + c];
  } else if (K == 1) {
    std::fill_n(g, (A * C), 0.);
    for (int h = 0; h < H; h++)
      for (int i = 0; i < A; i++)
        g[i] += M[h] * ea[h * A + i];
  } else {
    double U[H * C]{};
    std::fill_n(g, A * C, 0.);
    if constexpr (K == 4) {
      for (int h = 0; h < H; h++)
        for (int c = 0; c < C; c++)
          U[h * C + c] = M[h * K] * ec[c] + M[h * K + 3 - c] * ec[(3 - c) * 3 + c];
    } else {
      for (int h = 0; h < H; h++)
        for (int k = 0; k < K; k++) {
          double m = M[h * K + k];
          for (int c = 0; c < C; c++)
            U[h * C + c] += m * ec[k * C + c];
        }
    }
    if constexpr (H == 4) {
      for (int c = 0; c < C; c++)
        for (int i = 0; i < A; i++)
          g[i + A * c] = ea[i] * U[c] + ea[(3 - i) * 3 + i] * U[(3 - i) * C + c];
    } else {
      for (int h = 0; h < H; h++)
        for (int c = 0; c < C; c++) {
          double u = U[h * C + c];
          for (int i = 0; i < A; i++)
            g[i + A * c] += ea[h * A + i] * u;
        }
    }
  }
  for (size_t i = 0; i < (A * C); i++) {
    if (!std::isfinite(g[i]))
      return false;
    G[i] = g[i];
  }
  return true;
}

using LowFunction = bool (*)(const LowContext &, size_t, size_t, Real *, double);
static LowFunction select_low(const Layout &l) {
  if (l.hab == 1 && l.hcd == 1)
    return primitive_low_fixed<1, 1, 1, 1, 0>;
  if (l.hab == 1 && l.hcd == 4)
    return primitive_low_fixed<1, 4, 1, 3, 1>;
  if (l.hab == 4 && l.hcd == 1)
    return primitive_low_fixed<4, 1, 3, 1, 1>;
  if (l.hab == 4 && l.hcd == 4)
    return primitive_low_fixed<4, 4, 3, 3, 2>;
  if (l.hab == 1 && l.hcd == 10) {
    if (l.ncd == 5)
      return primitive_low_fixed<1, 10, 1, 5, 2>;
    if (l.ncd == 6)
      return primitive_low_fixed<1, 10, 1, 6, 2>;
    if (l.ncd == 9)
      return primitive_low_fixed<1, 10, 1, 9, 2>;
  }
  if (l.hab == 10 && l.hcd == 1) {
    if (l.nab == 5)
      return primitive_low_fixed<10, 1, 5, 1, 2>;
    if (l.nab == 6)
      return primitive_low_fixed<10, 1, 6, 1, 2>;
    if (l.nab == 9)
      return primitive_low_fixed<10, 1, 9, 1, 2>;
  }
  if (l.hab == 1 && l.hcd == 20) {
    if (l.ncd == 15)
      return primitive_low_fixed<1, 20, 1, 15, 3>;
    if (l.ncd == 18)
      return primitive_low_fixed<1, 20, 1, 18, 3>;
  }
  if (l.hab == 20 && l.hcd == 1) {
    if (l.nab == 15)
      return primitive_low_fixed<20, 1, 15, 1, 3>;
    if (l.nab == 18)
      return primitive_low_fixed<20, 1, 18, 1, 3>;
  }
  if (l.hab == 4 && l.hcd == 10) {
    if (l.ncd == 5)
      return primitive_low_fixed<4, 10, 3, 5, 3>;
    if (l.ncd == 6)
      return primitive_low_fixed<4, 10, 3, 6, 3>;
    if (l.ncd == 9)
      return primitive_low_fixed<4, 10, 3, 9, 3>;
  }
  if (l.hab == 10 && l.hcd == 4) {
    if (l.nab == 5)
      return primitive_low_fixed<10, 4, 5, 3, 3>;
    if (l.nab == 6)
      return primitive_low_fixed<10, 4, 6, 3, 3>;
    if (l.nab == 9)
      return primitive_low_fixed<10, 4, 9, 3, 3>;
  }
  return nullptr;
}

static bool compute_projected(const libgto_pair &a, const libgto_pair &b,
                              const libgto_eri_options &o, libgto_workspace &w, double *out,
                              const Layout &l, bool use_fp64, bool use_low = false,
                              bool use_dd = false) {
  w.stats = {};
  auto low_function = use_low ? select_low(l) : nullptr;
  if (use_low && !low_function)
    return false;
  LowContext context{};
  if (use_low)
    context = low_context(a, b, o.representation == LIBGTO_PURE);
  Real *R = w.arena.data();
  Real *M = R + l.r;
  Real *U = M + l.m;
  Real *G = U + l.u;
  Real *tmp = G + l.g;
  Real *contracted = tmp + l.tmp;
  Real *stage = contracted + l.contracted;
  std::fill_n(contracted, l.contracted, 0);
  size_t P = a.primitives.size(), Q = b.primitives.size();
  double mass = a.weight_mass * b.weight_mass;
  double monitor_budget = std::isnormal(mass) ? 0.0625e-10 / mass : 0;
  if (o.representation == LIBGTO_CARTESIAN)
    for (const Shell *s : {&a.a, &a.b, &b.a, &b.b})
      if (s->norm == LIBGTO_NORMALIZED_PRIMITIVES && s->l == 2)
        monitor_budget /= std::sqrt(3.);

  const bool legacy_ordered = l.L == 5 &&
                              ((l.nab == 9 && l.ncd == 15) || (l.nab == 15 && l.ncd == 9)) &&
                              (1 == 1 || (1 == 3 && a.a.r != a.b.r && b.a.r != b.b.r));

  const bool scoped_shape = (l.hab == 20 && l.hcd == 20) ||
                            (l.hab == 10 && l.nab == 9 && l.hcd == 35) ||
                            (l.hab == 35 && l.hcd == 10 && l.ncd == 9) ||
                            (l.hab == 20 && l.hcd == 35) || (l.hab == 35 && l.hcd == 20);
  const bool added_ordered = (3 == 1 && l.L == 6 && l.nab == 15 && l.ncd == 15) ||
                             (3 == 2 && l.L >= 5 && l.L <= 7 && l.hab > 1 && l.hcd > 1) ||
                             (3 == 3 && scoped_shape);
  const bool use_added_ordered = added_ordered && ordered_environment();
  const bool dsdp_shape = (l.hab == 10 && l.nab == 5 && l.hcd == 20 && l.ncd == 15) ||
                          (l.hab == 20 && l.nab == 15 && l.hcd == 10 && l.ncd == 5);
  const bool dsdp_ordered = dsdp_shape && a.a.r != a.b.r && b.a.r != b.b.r && ordered_environment();
  const bool ordered_angular = legacy_ordered || use_added_ordered || dsdp_ordered;

  const bool ps_shape = l.hab == 4 && l.nab == 3 && l.hcd == 35 && l.ncd == 25;
  const bool ds_shape = l.hab == 10 && l.nab == 5 &&
                        ((l.hcd == 35 && l.ncd == 25) ||
                         (l.hcd == 20 && l.ncd == 15 && (a.a.r == a.b.r || b.a.r == b.b.r)));
  const bool ket_shape = l.hab == 35 && l.nab == 25 && l.hcd == 4 && l.ncd == 3;
  const bool ps_bra = (((7 & 1) && ps_shape) || ((7 & 2) && ds_shape) || ((7 & 4) && ket_shape)) &&
                      o.representation == LIBGTO_PURE && ordered_environment();
  const bool defer_clear = ordered_angular && 1;
  bool local_wide = false;
  const bool uniform_enabled = 1 && l.L >= 5 && l.L <= 7 && ordered_environment();
  double uniform_error = std::numeric_limits<double>::quiet_NaN();
  auto evaluate = [&](size_t p, size_t q) {
    double t = o.profile ? now() : 0;
    bool success = true;
    if (use_low) {
      ++w.precision.fp64_primitive_quartets;
      success = low_function(context, p, q, G, monitor_budget);
      if (!success) {
        // Each accepted primitive obeys the unchanged absolute budget. A
        // rejected primitive is evaluated wide, preserving the channel sum
        // order instead of discarding all preceding accepted primitives.
        if (!local_wide) {
          ++w.precision.fp64_fallbacks;
          ++w.precision.wide_calls;
          local_wide = true;
        }
        primitive(a, b, p, q, l, R, M, U, G, o.representation == LIBGTO_PURE);
        ++w.stats.primitive_quartets;
        success = true;
      }
    } else if (use_dd) {
      double gd[25], ge[25];
      ++w.precision.fp64_primitive_quartets;
      if (primitive_dd_moment(a, b, p, q, l, gd, ge, monitor_budget)) {
        for (int i = 0; i < 25; i++)
          G[i] = Real(gd[i]);
      } else {
        if (!local_wide) {
          ++w.precision.fp64_fallbacks;
          ++w.precision.wide_calls;
          local_wide = true;
        }
        primitive(a, b, p, q, l, R, M, U, G, true);
        ++w.stats.primitive_quartets;
      }
    } else if (use_fp64) {
      ++w.precision.fp64_primitive_quartets;
      success = primitive_fp64(a, b, p, q, l, R, M, w.fast.data(), o.representation == LIBGTO_PURE,
                               monitor_budget, ordered_angular,
                               uniform_enabled ? &uniform_error : nullptr, defer_clear, ps_bra);
    } else
      primitive(a, b, p, q, l, R, M, U, G, o.representation == LIBGTO_PURE);
    if (o.profile)
      w.stats.recurrence_seconds += now() - t;
    ++w.stats.primitive_quartets;
    return success;
  };
  if (use_fp64) {
    double *gd = w.fast.data() + 2 * l.u;
    double *ge = gd + l.g;
    double *td = gd + 3 * l.g;
    double *te = td + l.tmp;
    double *od = te + l.tmp;
    double *oe = od + l.contracted;
    constexpr double eps = std::numeric_limits<double>::epsilon();
    std::fill_n(od, l.contracted, 0.);
    std::fill_n(oe, l.contracted, 0.);
    if (l.cab == 1 && l.ccd == 1) {
      for (size_t q = 0; q < Q; q++)
        for (size_t p = 0; p < P; p++) {
          if (!evaluate(p, q))
            return false;
          double clock = o.profile ? now() : 0;
          Real wr = a.weights[p] * b.weights[q];
          if (std::abs(wr) > std::numeric_limits<double>::max())
            return false;
          double weight = static_cast<double>(wr);
          if (wr != 0 && !std::isnormal(weight))
            return false;
          if (std::isfinite(uniform_error))
            update_channels<true>(l.g, weight, gd, &uniform_error, od, oe);
          else
            update_channels<false>(l.g, weight, gd, ge, od, oe);
          if (o.profile)
            w.stats.contraction_seconds += now() - clock;
        }
    } else {
      size_t outer = l.abfirst ? Q : P, inner = l.abfirst ? P : Q;
      size_t ni = l.abfirst ? l.cab : l.ccd, no = l.abfirst ? l.ccd : l.cab;
      const auto &wi = l.abfirst ? a.weights64 : b.weights64;
      const auto &wo = l.abfirst ? b.weights64 : a.weights64;
      for (size_t q = 0; q < outer; q++) {
        std::fill_n(td, l.tmp, 0.);
        std::fill_n(te, l.tmp, 0.);
        for (size_t p = 0; p < inner; p++) {
          if (!evaluate(l.abfirst ? p : q, l.abfirst ? q : p))
            return false;
          double clock = o.profile ? now() : 0;
          for (size_t c = 0; c < ni; c++) {
            double weight = wi[p + inner * c];
            if (std::isfinite(uniform_error))
              update_channels<true>(l.g, weight, gd, &uniform_error, td + l.g * c, te + l.g * c);
            else
              update_channels<false>(l.g, weight, gd, ge, td + l.g * c, te + l.g * c);
          }
          if (o.profile)
            w.stats.contraction_seconds += now() - clock;
        }
        double clock = o.profile ? now() : 0;
        for (size_t d = 0; d < no; d++)
          for (size_t c = 0; c < ni; c++) {
            double weight = wo[q + outer * d];
            size_t slot = l.abfirst ? c + l.cab * d : d + l.cab * c;
            for (size_t v = 0; v < l.g; v++) {
              size_t src = v + l.g * c, dst = v + l.g * slot;
              double product = weight * td[src];
              od[dst] += product;
              oe[dst] += std::abs(weight) * te[src] + 2 * eps * std::abs(product) +
                         eps * std::abs(od[dst]);
            }
          }
        if (o.profile)
          w.stats.contraction_seconds += now() - clock;
      }
    }
    double scale = 1;
    if (o.representation == LIBGTO_CARTESIAN)
      for (const Shell *s : {&a.a, &a.b, &b.a, &b.b})
        if (s->norm == LIBGTO_NORMALIZED_PRIMITIVES && s->l == 2)
          scale *= std::sqrt(3.);
    for (size_t i = 0; i < l.contracted; i++) {
      double bound = oe[i] * (1 + 4 * eps) + std::numeric_limits<double>::denorm_min();
      if (!std::isfinite(od[i]) || !std::isfinite(bound) ||
          scale * bound > .25 * (1e-10 + 1e-11 * std::abs(od[i])))
        return false;
      contracted[i] = od[i];
    }
  } else if (l.cab == 1 && l.ccd == 1) {
    for (size_t q = 0; q < Q; ++q) {
      for (size_t p = 0; p < P; ++p) {
        if (!evaluate(p, q))
          return false;
        double t = o.profile ? now() : 0;
        Real weight = a.weights[p] * b.weights[q];
        for (size_t v = 0; v < l.g; v++) {
          contracted[v] += weight * G[v];
        }
        if (o.profile)
          w.stats.contraction_seconds += now() - t;
      }
    }
  } else {
    size_t outer = l.abfirst ? Q : P, inner = l.abfirst ? P : Q;
    size_t ninner = l.abfirst ? l.cab : l.ccd, nouter = l.abfirst ? l.ccd : l.cab;
    const auto &wi = l.abfirst ? a.weights : b.weights;
    const auto &wo = l.abfirst ? b.weights : a.weights;
    for (size_t q = 0; q < outer; ++q) {
      std::fill_n(tmp, l.tmp, 0);
      for (size_t p = 0; p < inner; ++p) {
        if (!evaluate(l.abfirst ? p : q, l.abfirst ? q : p))
          return false;
        double t = o.profile ? now() : 0;
        for (size_t c = 0; c < ninner; ++c) {
          Real weight = wi[p + inner * c];
          for (size_t v = 0; v < l.g; v++) {
            tmp[v + l.g * c] += weight * G[v];
          }
        }
        if (o.profile)
          w.stats.contraction_seconds += now() - t;
      }
      double t = o.profile ? now() : 0;
      for (size_t d = 0; d < nouter; ++d) {
        for (size_t c = 0; c < ninner; ++c) {
          Real weight = wo[q + outer * d];
          size_t slot = l.abfirst ? c + l.cab * d : d + l.cab * c;
          for (size_t v = 0; v < l.g; ++v) {
            contracted[v + l.g * slot] += weight * tmp[v + l.g * c];
          }
        }
      }
      if (o.profile)
        w.stats.contraction_seconds += now() - t;
    }
  }
  return store_md_eri_output(a, b, o, w, contracted, stage, out, l);
}

using LowDoubleFunction = bool (*)(const LowContext &, size_t, size_t, double *, double);
static LowDoubleFunction select_low_double(const Layout &l) {
  if (l.hab == 1 && l.hcd == 1)
    return primitive_low_fixed<1, 1, 1, 1, 0, double>;
  if (l.hab == 1 && l.hcd == 4)
    return primitive_low_fixed<1, 4, 1, 3, 1, double>;
  if (l.hab == 4 && l.hcd == 1)
    return primitive_low_fixed<4, 1, 3, 1, 1, double>;
  if (l.hab == 4 && l.hcd == 4)
    return primitive_low_fixed<4, 4, 3, 3, 2, double>;
  if (l.hab == 1 && l.hcd == 10) {
    if (l.ncd == 5)
      return primitive_low_fixed<1, 10, 1, 5, 2, double>;
    if (l.ncd == 6)
      return primitive_low_fixed<1, 10, 1, 6, 2, double>;
    if (l.ncd == 9)
      return primitive_low_fixed<1, 10, 1, 9, 2, double>;
  }
  if (l.hab == 10 && l.hcd == 1) {
    if (l.nab == 5)
      return primitive_low_fixed<10, 1, 5, 1, 2, double>;
    if (l.nab == 6)
      return primitive_low_fixed<10, 1, 6, 1, 2, double>;
    if (l.nab == 9)
      return primitive_low_fixed<10, 1, 9, 1, 2, double>;
  }
  if (l.hab == 1 && l.hcd == 20) {
    if (l.ncd == 15)
      return primitive_low_fixed<1, 20, 1, 15, 3, double>;
    if (l.ncd == 18)
      return primitive_low_fixed<1, 20, 1, 18, 3, double>;
  }
  if (l.hab == 20 && l.hcd == 1) {
    if (l.nab == 15)
      return primitive_low_fixed<20, 1, 15, 1, 3, double>;
    if (l.nab == 18)
      return primitive_low_fixed<20, 1, 18, 1, 3, double>;
  }
  if (l.hab == 4 && l.hcd == 10) {
    if (l.ncd == 5)
      return primitive_low_fixed<4, 10, 3, 5, 3, double>;
    if (l.ncd == 6)
      return primitive_low_fixed<4, 10, 3, 6, 3, double>;
    if (l.ncd == 9)
      return primitive_low_fixed<4, 10, 3, 9, 3, double>;
  }
  if (l.hab == 10 && l.hcd == 4) {
    if (l.nab == 5)
      return primitive_low_fixed<10, 4, 5, 3, 3, double>;
    if (l.nab == 6)
      return primitive_low_fixed<10, 4, 6, 3, 3, double>;
    if (l.nab == 9)
      return primitive_low_fixed<10, 4, 9, 3, 3, double>;
  }
  return nullptr;
}

// Accepted low-angular primitive values are exactly binary64 already. Keep
// that storage representation until promotion in the unchanged wide product.
// Rejected primitive blocks retain their full wide result, without a cast.
template <class Input>
static inline void wide_update(size_t n, Real weight, const Input *src, Real *dst) {
  for (size_t v = 0; v < n; v++)
    dst[v] += weight * src[v];
}
#include "scaled_low.hpp"
#include "general_zero_packet.hpp"
#include "general_single_scaled.hpp"
#include "two_p_packing.hpp"
#include "general_two_inner64.hpp"
#include "general_single_inner64.hpp"
#include "general_two_moment.hpp"
#include "general_three_moment_stable.hpp"
#include "general_three_moment_symmetric.hpp"
#include "general_dplusp.hpp"
#include "general_same_side.hpp"

template <int Scale>
static bool compute_low_storage_impl(const libgto_pair &a, const libgto_pair &b,
                                     const libgto_eri_options &o, libgto_workspace &w, double *out,
                                     const Layout &l) {
  auto low = select_low_double(l);
  if (!low)
    return false;
  w.stats = {};
  const LowContext context = low_context(a, b, o.representation == LIBGTO_PURE);
  Real *R = w.arena.data();
  Real *M = R + l.r;
  Real *U = M + l.m;
  Real *G = U + l.u;
  Real *tmp = G + l.g;
  Real *contracted = tmp + l.tmp;
  Real *stage = contracted + l.contracted;
  std::fill_n(contracted, l.contracted, 0);
  double gd[32];
  size_t P = a.primitives.size(), Q = b.primitives.size();
  double mass = a.weight_mass * b.weight_mass, budget = std::isnormal(mass) ? 0.0625e-10 / mass : 0;
  if (o.representation == LIBGTO_CARTESIAN)
    for (const Shell *x : {&a.a, &a.b, &b.a, &b.b})
      if (x->norm == LIBGTO_NORMALIZED_PRIMITIVES && x->l == 2)
        budget /= std::sqrt(3.);
  bool local_wide = false;
  auto evaluate = [&](size_t p, size_t q) {
    double t = o.profile ? now() : 0;
    ++w.precision.fp64_primitive_quartets;
    bool narrow = low(context, p, q, gd, budget);
    if (!narrow) {
      bool recovered = false;
      if constexpr (Scale == 1)
        recovered = compute_scaled_scalar(context, p, q, G, budget);
      if constexpr (Scale == 2)
        recovered = compute_scaled_low<false>(context, p, q, G, budget);
      if constexpr (Scale == 4)
        recovered = compute_scaled_low<true>(context, p, q, G, budget, a.a.l + a.b.l == 1);
      if (recovered) {
        ++w.stats.primitive_quartets;
        return false;
      }
      if (!local_wide) {
        ++w.precision.fp64_fallbacks;
        ++w.precision.wide_calls;
        local_wide = true;
      }
      primitive(a, b, p, q, l, R, M, U, G, o.representation == LIBGTO_PURE);
      ++w.stats.primitive_quartets;
    }
    if (o.profile)
      w.stats.recurrence_seconds += now() - t;
    ++w.stats.primitive_quartets;
    return narrow;
  };
  if (l.cab == 1 && l.ccd == 1) {
    for (size_t q = 0; q < Q; q++)
      for (size_t p = 0; p < P; p++) {
        bool narrow = evaluate(p, q);
        double t = o.profile ? now() : 0;
        Real weight = a.weights[p] * b.weights[q];
        if (narrow)
          wide_update(l.g, weight, gd, contracted);
        else
          wide_update(l.g, weight, G, contracted);
        if (o.profile)
          w.stats.contraction_seconds += now() - t;
      }
  } else {
    size_t outer = l.abfirst ? Q : P, inner = l.abfirst ? P : Q, ni = l.abfirst ? l.cab : l.ccd,
           no = l.abfirst ? l.ccd : l.cab;
    const auto &wi = l.abfirst ? a.weights : b.weights;
    const auto &wo = l.abfirst ? b.weights : a.weights;
    for (size_t q = 0; q < outer; q++) {
      std::fill_n(tmp, l.tmp, 0);
      for (size_t p = 0; p < inner; p++) {
        bool narrow = evaluate(l.abfirst ? p : q, l.abfirst ? q : p);
        double t = o.profile ? now() : 0;
        if (narrow) {
          for (size_t c = 0; c < ni; c++)
            wide_update(l.g, wi[p + inner * c], gd, tmp + l.g * c);
        } else {
          for (size_t c = 0; c < ni; c++)
            wide_update(l.g, wi[p + inner * c], G, tmp + l.g * c);
        }
        if (o.profile)
          w.stats.contraction_seconds += now() - t;
      }
      double t = o.profile ? now() : 0;
      for (size_t d = 0; d < no; d++)
        for (size_t c = 0; c < ni; c++) {
          size_t slot = l.abfirst ? c + l.cab * d : d + l.cab * c;
          wide_update(l.g, wo[q + outer * d], tmp + l.g * c, contracted + l.g * slot);
        }
      if (o.profile)
        w.stats.contraction_seconds += now() - t;
    }
  }
  return store_md_eri_output(a, b, o, w, contracted, stage, out, l);
}
static bool compute_low_storage(const libgto_pair &a, const libgto_pair &b,
                                const libgto_eri_options &o, libgto_workspace &w, double *out,
                                const Layout &l) {
  if (!o.profile) {
    if (l.L == 0) {
      return compute_low_storage_impl<2>(a, b, o, w, out, l);
    }
    if (l.L == 1)
      return compute_low_storage_impl<4>(a, b, o, w, out, l);
  }
  return compute_low_storage_impl<0>(a, b, o, w, out, l);
}
static void compute_impl(const libgto_pair &a, const libgto_pair &b, const libgto_eri_options &o,
                         libgto_workspace &w, double *out, const Layout &l) {
  if (w.policy == LIBGTO_PRECISION_REFERENCE) {
    ++w.precision.wide_calls;
    auto ref = layout_reference(a, b, o);
    compute_reference(a, b, o, w, out, ref);
    return;
  }
  // Small dense blocks use the readable wide projected path when monitoring
  // and packing would dominate. This is a cost policy, not a screening rule.
  // The two dense loops vectorize over ncd and nab, respectively.
  // Short axes do not amortize FP64 monitoring and packing in measured cases.
  // This empirical cost cutoff changes precision policy, never output work.
  if (l.L <= 3 && a.fp64_safe && b.fp64_safe) {
    auto before = w.precision.fp64_fallbacks;
    if (l.L == 0) {
      bool done = compute_scalar_panel<false>(a, b, o, w, out, l);
      if (done) {
        if (w.precision.fp64_fallbacks == before)
          ++w.precision.fp64_calls;
        return;
      }
    }
    if (l.L == 2 && compute_same_pair_panel(a, b, o, w, out, l)) {
      if (w.precision.fp64_fallbacks == before)
        ++w.precision.fp64_calls;
      return;
    }
    if (l.L == 3) {
      if (compute_quadratic_linear_panel(a, b, o, w, out, l)) {
        if (w.precision.fp64_fallbacks == before)
          ++w.precision.fp64_calls;
        return;
      }
      if (compute_oriented_three_p_panel(a, b, o, w, out, l)) {
        if (w.precision.fp64_fallbacks == before)
          ++w.precision.fp64_calls;
        return;
      }
    }

    {
      bool done = false;
      uint64_t extra = 0;
      if (single_p_panel_eligible(a, b, o, l)) {
        done = contract_single_p_panel(a, b, o, w, out, l);
        if (!done) {
          extra = w.stats.primitive_quartets;
        }
      }
      if (!done)
        done = compute_single_p_panel(a, b, o, w, out, l);
      w.stats.primitive_quartets += extra;
      if (done) {
        if (w.precision.fp64_fallbacks == before)
          ++w.precision.fp64_calls;
        return;
      }
    }
    {
      bool done = false;
      uint64_t extra = 0;
      if (two_p_panel_eligible(a, b, o, l)) {
        done = contract_two_p_panel(a, b, o, w, out, l);
        if (!done) {
          extra = w.stats.primitive_quartets;
        }
      }
      if (!done)
        done = compute_two_p_moment_panel(a, b, o, w, out, l);
      w.stats.primitive_quartets += extra;
      if (done) {
        if (w.precision.fp64_fallbacks == before)
          ++w.precision.fp64_calls;
        return;
      }
    }
    if (compute_low_storage(a, b, o, w, out, l)) {
      if (w.precision.fp64_fallbacks == before)
        ++w.precision.fp64_calls;
      return;
    }
    ++w.precision.fp64_fallbacks;
    ++w.precision.wide_calls;
    auto partial = w.stats;
    require(compute_projected(a, b, o, w, out, l, false), LIBGTO_NUMERICAL_FAILURE);
    w.stats.primitive_quartets += partial.primitive_quartets;
    w.stats.recurrence_seconds += partial.recurrence_seconds;
    w.stats.contraction_seconds += partial.contraction_seconds;
    return;
  }

  bool ddshape = o.representation == LIBGTO_PURE && l.L == 4 && l.hab == 10 && l.hcd == 10 &&
                 l.nab == 5 && l.ncd == 5;

  bool fast = (l.g > 1 && (l.nab >= 9 || l.ncd >= 9)) || (ddshape && (2 == 1 || 2 == 2));
  if (fast && (!a.fp64_safe || !b.fp64_safe)) {
    ++w.precision.fp64_fallbacks;
    ++w.precision.wide_calls;
    require(compute_projected(a, b, o, w, out, l, false), LIBGTO_NUMERICAL_FAILURE);
    return;
  }
  if (compute_projected(a, b, o, w, out, l, fast)) {
    if (fast)
      ++w.precision.fp64_calls;
    else
      ++w.precision.wide_calls;
    return;
  }
  ++w.precision.fp64_fallbacks;
  ++w.precision.wide_calls;
  libgto_statistics partial = w.stats;
  require(compute_projected(a, b, o, w, out, l, false), LIBGTO_NUMERICAL_FAILURE);
  w.stats.primitive_quartets += partial.primitive_quartets;
  w.stats.recurrence_seconds += partial.recurrence_seconds;
  w.stats.contraction_seconds += partial.contraction_seconds;
  w.stats.transform_seconds += partial.transform_seconds;
}
void compute_bound(const libgto_pair &a, const libgto_pair &b, const libgto_eri_options &o,
                   libgto_workspace &w, double *out, const Layout &l, bool requires_raw) {
  if (high_angular(a, b)) {
    compute_high(a, b, o, w, out, l);
    return;
  }
  if (w.policy == LIBGTO_PRECISION_REFERENCE) {
    compute_impl(a, b, o, w, out, l);
    return;
  }
  if (requires_raw) {
    compute_sensitive(a, b, o, w, out, l);
    return;
  }
  if ((l.L & 1) && a.a.r == a.b.r && a.a.r == b.a.r && a.a.r == b.b.r) {
    std::fill_n(out, l.public_size.elements, 0.);
    w.stats = {};
    ++w.precision.wide_calls;
    return;
  }
  // Layout/output dimensions and contraction direction stay those of the parent.
  // Only the primitive geometry summation indices are reduced, with exact factors.
  const auto &aa = a.optimized ? *a.optimized : a;
  const auto &bb = b.optimized ? *b.optimized : b;
  compute_impl(aa, bb, o, w, out, l);
}
void compute(const libgto_pair &a, const libgto_pair &b, const libgto_eri_options &o,
             libgto_workspace &w, double *out, const Layout &l) {
  compute_bound(a, b, o, w, out, l, needs_sensitive(a, b, l, w.policy));
}

}
