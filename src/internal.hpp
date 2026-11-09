#ifndef LIBGTO_INTERNAL_HPP
#define LIBGTO_INTERNAL_HPP
#include "libgto.h"
#include <array>
#include <memory>
#include <new>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <chrono>
namespace gto {
using Real = long double;
constexpr Real pi = 3.141592653589793238462643383279502884L;
struct Error {
  libgto_status status;
};
inline void require(bool ok, libgto_status s = LIBGTO_INVALID_ARGUMENT) {
  if (!ok)
    throw Error{s};
}
inline size_t mul(size_t a, size_t b) {
  require(!b || a <= std::numeric_limits<size_t>::max() / b);
  return a * b;
}
inline size_t add(size_t a, size_t b) {
  require(a <= std::numeric_limits<size_t>::max() - b);
  return a + b;
}
inline int ncart(int l) {
  return (l + 1) * (l + 2) / 2;
}
constexpr int nh(int l) {
  return (l + 1) * (l + 2) * (l + 3) / 6;
}
constexpr int hi(int t, int u, int v) {
  int d = t + u + v;
  return d * (d + 1) * (d + 2) / 6 + t * (d + 1) - t * (t - 1) / 2 + u;
}
using Triple = std::array<int, 3>;
std::vector<Triple> orbitals(int);
std::vector<Triple> hermites(int);
void boys(unsigned, double, double *);
// Mathematical data container; current angular capability is an MD preparation check.
struct Shell {
  int l;
  std::array<Real, 3> r;
  size_t k, c;
  libgto_normalization norm;
  std::vector<Real> a, weights;
  std::vector<double> raw_coefficients;
};
struct Primitive {
  Real p, K;
  std::array<Real, 3> offset; /* Gaussian product center minus first shell center. */
  double p64 = 0, K64 = 0;
  bool low_normal = false;
};
// Concrete current MD execution plan, not a public basis or operator representation.
struct Layout {
  size_t r, m, u, g, tmp, contracted, stage, borrowed, total, fast;
  int L, hab, hcd, nab, ncd;
  size_t cab, ccd;
  bool abfirst;
  libgto_requirements public_size;
};
// Extra raw high-precision demand is distinct from reusable base arenas.
inline size_t layout_payload(const Layout &l) {
  return add(mul(l.total, sizeof(Real)), mul(add(l.borrowed, l.fast), sizeof(double)));
}
inline size_t high_workspace_extra(const Layout &l) {
  size_t base = layout_payload(l);
  require(l.public_size.workspace_bytes >= base);
  return l.public_size.workspace_bytes - base;
}
// Mathematical semantics and shape are independent of the prepared pair identities.
enum class OperatorKind { Coulomb };
struct MathRequest {
  OperatorKind op = OperatorKind::Coulomb;
  unsigned rank = 4, derivative_order = 0;
  libgto_representation representation = LIBGTO_CARTESIAN;
  libgto_precision_policy precision = LIBGTO_PRECISION_AUTO;
  libgto_contraction_order contraction_order = LIBGTO_CONTRACT_AUTO;
};
struct QuartetShape {
  std::array<int, 4> l{};
  std::array<size_t, 4> k{}, c{};
  std::array<libgto_normalization, 4> normalization{};
};
struct FamilyPlan {
  MathRequest request;
  QuartetShape shape;
  Layout storage;
};
struct PlanKey {
  std::array<size_t, 8> sizes{};
  unsigned semantics = 0;
  bool operator==(const PlanKey &b) const {
    return semantics == b.semantics && sizes == b.sizes;
  }
};
struct PlanEntry {
  PlanKey key;
  FamilyPlan plan;
  bool valid = false;
};
struct BlockBinding {
  const libgto_eri_task *task = nullptr;
  const FamilyPlan *plan = nullptr;
};
struct OutputRange {
  uintptr_t begin, end;
};
const FamilyPlan *cached_plan(libgto_batch &, const libgto_eri_task &, libgto_precision_policy);

void require_md_angular_momentum(int);
bool high_angular(const libgto_pair &, const libgto_pair &);
void prepare_high_pair(libgto_pair &);
void rounded_view(const std::vector<Real> &, std::vector<double> &, bool &);
void compute_high(const libgto_pair &, const libgto_pair &, const libgto_eri_options &,
                  libgto_workspace &, double *, const Layout &);
size_t high_shape_mp_bytes(const QuartetShape &, const Layout &);
size_t high_mp_bytes(const libgto_pair &, const libgto_pair &, const Layout &);
QuartetShape quartet_shape(const libgto_pair &, const libgto_pair &);
MathRequest eri_request(const libgto_eri_options &, libgto_precision_policy);
FamilyPlan resolve_plan(const MathRequest &, const QuartetShape &);
Layout layout_reference(const libgto_pair &, const libgto_pair &, const libgto_eri_options &);
Layout layout(const libgto_pair &, const libgto_pair &, const libgto_eri_options &,
              libgto_precision_policy = LIBGTO_PRECISION_AUTO);
struct HighScratch;
size_t high_scratch_bytes(const libgto_workspace &) noexcept;
struct SensitiveScratch;
struct SensitiveDeleter {
  void operator()(SensitiveScratch *) const noexcept;
};
size_t sensitive_workspace_bytes() noexcept;
void reserve_sensitive(libgto_workspace &);
void reserve(libgto_workspace &, size_t, size_t, size_t, bool precise = false);
bool store_md_eri_output(const libgto_pair &, const libgto_pair &, const libgto_eri_options &,
                         libgto_workspace &, const Real *, Real *, double *, const Layout &);
void compute_reference(const libgto_pair &, const libgto_pair &, const libgto_eri_options &,
                       libgto_workspace &, double *, const Layout &);
void compute_sensitive(const libgto_pair &, const libgto_pair &, const libgto_eri_options &,
                       libgto_workspace &, double *, const Layout &);
void compute_bound(const libgto_pair &, const libgto_pair &, const libgto_eri_options &,
                   libgto_workspace &, double *, const Layout &, bool requires_raw);
void compute(const libgto_pair &, const libgto_pair &, const libgto_eri_options &,
             libgto_workspace &, double *, const Layout &);
// MD preparation/plan/execution services; public compatibility wrappers stay in api.cpp.
std::unique_ptr<libgto_pair> make_pair(const libgto_shell &, const libgto_shell &);
void execute_batch(libgto_batch *, const libgto_eri_task *, size_t);
template <class F> libgto_status protect(F f) {
  try {
    f();
    return LIBGTO_SUCCESS;
  } catch (const Error &e) {
    return e.status;
  } catch (const std::bad_alloc &) {
    return LIBGTO_ALLOCATION_FAILED;
  } catch (...) {
    return LIBGTO_NUMERICAL_FAILURE;
  }
}
inline double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}
// Private bounded numeric views, owned by the immutable pair. Not a public layout.
namespace gto {
struct PairNumerics {
  bool common_low = false;
  unsigned low_degree = 0;
  std::array<Real, 3> common_offset{};
  Real active = 0, pmin = 0, sqrt_pmax = 0, m1 = 0, m125 = 0, m15 = 0;
};
}
struct libgto_pair {
  std::unique_ptr<libgto_pair> optimized;
  gto::Shell a, b;
  int H, N;
  std::array<size_t, 4> shape_sizes{};
  unsigned shape_bits = 0;
  bool sensitive = false;
  size_t shape_hash = 0;
  std::vector<gto::Primitive> primitives;
  std::vector<gto::Real> E, pure_E, weights;
  std::vector<uint64_t> E_mask, pure_mask;
  bool sparse_cart = false, sparse_pure = false;
  double weight_mass = 0;
  std::vector<double> norm_cart, norm_pure;
  std::vector<double> E64, pure_E64, weights64;
  bool fp64_safe = true; /* E[h*N+cartPair], weights[primitive+P*channel] */
  gto::PairNumerics numerics;
};
struct libgto_workspace {
  size_t capacity_limit = std::numeric_limits<size_t>::max();
  std::shared_ptr<gto::HighScratch> high;
  std::unique_ptr<gto::SensitiveScratch, gto::SensitiveDeleter> precise;
  uint64_t sensitive_calls = 0;
  std::vector<gto::Real> arena;
  std::vector<double> borrowed, fast;
  libgto_statistics stats{};
  libgto_precision_statistics precision{};
  libgto_precision_policy policy = LIBGTO_PRECISION_AUTO;
};
struct libgto_batch {
  libgto_batch_options options;
  std::vector<libgto_workspace> work;
  std::array<gto::PlanEntry, 256> plans{};
  std::array<unsigned short, 512> plan_index{};
  size_t plan_count = 0;
  std::vector<gto::Layout> parallel_layouts;
  std::vector<gto::BlockBinding> bindings;
  std::vector<gto::OutputRange> ranges;
  size_t active_bindings = 0;
  size_t binding_limit = 2048, binding_byte_limit = 1024 * 1024;
  libgto_binding_statistics binding_stats{};
};

namespace gto {

inline bool prepared_angular_risk(const libgto_pair &a, const libgto_pair &b) {
  {
    {}
  }
  const auto &x = a.numerics;
  const auto &y = b.numerics;
  if (a.sensitive || b.sensitive)
    return false;
  if (x.low_degree == 2 && y.low_degree == 2) {
    // Empirical raw-input route for amplified, near-coincident pure quadratic
    // cancellation. Positive coefficient masses are prepared even when product
    // offsets differ. Use original-center distances, never a rounded small G.
    Real geom = 0;
    for (int d = 0; d < 3; d++)
      geom += std::abs(a.a.r[d] - a.b.r[d]) + std::abs(b.a.r[d] - b.b.r[d]) +
              std::abs(a.a.r[d] - b.a.r[d]);
    if (geom * std::max(x.sqrt_pmax, y.sqrt_pmax) > Real(1) / 16)
      return false;
    Real amp = std::min({x.m15 * y.m1, x.m1 * y.m15, x.m125 * y.m125 / std::sqrt(Real(2))});
    Real va = 1 / (2 * x.pmin), vc = 1 / (2 * y.pmin), cross = 1 / (2 * (x.pmin + y.pmin)),
         m = 2 * geom, mm = m * m;
    // Positive Gaussian fourth-moment envelope includes the isotropic terms
    // that can cancel in the prepared-wide Cartesian-to-pure construction.
    Real env = 36 * (mm + va) * (mm + vc) + 48 * cross * mm + 24 * cross * cross;
    Real risk =
        2 * pi * pi * std::sqrt(pi) * amp * env * (8192 * std::numeric_limits<Real>::epsilon());
    risk *= 1 + 64 * std::numeric_limits<double>::epsilon();
    return !std::isfinite(risk) || risk > 1e-11L; // Scoped empirical policy, NOT a certificate.
  }
  if (!x.common_low || !y.common_low)
    return false;
  unsigned degree = x.low_degree + y.low_degree;
  if (degree != 1 && degree != 2)
    return false;
  Real amp = std::min({x.m15 * y.m1, x.m1 * y.m15, x.m125 * y.m125 / std::sqrt(Real(2))});
  Real prefupper = 2 * pi * pi * std::sqrt(pi) * amp, risk;
  if (degree == 1) {
    // Wide product offsets may lose absolute information before a tiny odd moment
    // is formed. Estimate potential amplification from ORIGINAL coefficient mass.
    // Coordinate differences avoid an artificial absolute-origin dependence.
    Real geom = 0;
    for (int d = 0; d < 3; d++)
      geom += std::abs(a.a.r[d] - b.a.r[d]) + std::abs(x.common_offset[d]) +
              std::abs(y.common_offset[d]);
    Real active = x.low_degree ? x.active : y.active;
    Real scale = active + (1 + 2 * std::min(x.sqrt_pmax, y.sqrt_pmax) * active) * geom;
    risk = prefupper * scale * (4096 * std::numeric_limits<Real>::epsilon());
  } else {
    if (x.low_degree != 1 || y.low_degree != 1)
      return false;
    for (int d = 0; d < 3; d++)
      if ((a.a.r[d] - b.a.r[d]) + (x.common_offset[d] - y.common_offset[d]) != 0)
        return false;
    risk = prefupper * (x.active * y.active + 1 / (6 * (x.pmin + y.pmin))) *
           (1024 * std::numeric_limits<Real>::epsilon());
  }
  risk *= 1 + 64 * std::numeric_limits<double>::epsilon();
  return !std::isfinite(risk) || risk > 1e-11L; // Empirical risk, not an error certificate.
}

inline bool needs_sensitive(const libgto_pair &a, const libgto_pair &b, const Layout &l,
                            libgto_precision_policy p) {
  {
    {}
  }
  if (high_angular(a, b))
    return false;
  return p == LIBGTO_PRECISION_AUTO && l.L <= 8 &&
         (a.sensitive || b.sensitive || prepared_angular_risk(a, b));
}
}
#endif
