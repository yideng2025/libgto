#include "internal.hpp"
#include <map>
namespace gto {
static Shell copy_shell(const libgto_shell &v) {
  require_md_angular_momentum(v.angular_momentum);
  require(v.primitive_count && v.channel_count && v.exponents && v.coefficients);
  require(v.normalization == LIBGTO_UNNORMALIZED ||
          v.normalization == LIBGTO_NORMALIZED_PRIMITIVES);
  Shell s;
  s.l = v.angular_momentum;
  s.k = v.primitive_count;
  s.c = v.channel_count;
  s.norm = v.normalization;
  for (int i = 0; i < 3; i++) {
    require(std::isfinite(v.center[i]));
    s.r[i] = v.center[i];
  }
  mul(s.k, sizeof(double));
  mul(mul(s.k, s.c), sizeof(double));
  require(s.k <= s.a.max_size() && mul(s.k, s.c) <= s.weights.max_size());
  s.raw_coefficients.assign(v.coefficients, v.coefficients + mul(s.k, s.c));
  s.a.assign(v.exponents, v.exponents + s.k);
  s.weights.resize(mul(s.k, s.c));
  for (size_t p = 0; p < s.k; p++) {
    require(s.a[p] > 0 && std::isfinite(s.a[p]));
    Real norm = 1.;
    if (s.norm == LIBGTO_NORMALIZED_PRIMITIVES)
      norm = std::pow(2 * s.a[p] / pi, .75L) * std::pow(4 * s.a[p], .5L * s.l) / std::sqrt([&] {
               Real df = 1;
               for (int j = 2 * s.l - 1; j > 1; j -= 2)
                 df *= j;
               return df;
             }());
    require(std::isfinite(norm), LIBGTO_NUMERICAL_FAILURE);
    for (size_t c = 0; c < s.c; c++) {
      double w = v.coefficients[p + s.k * c];
      require(std::isfinite(w));
      s.weights[p + s.k * c] = w * norm;
      require(std::isfinite(w * norm), LIBGTO_NUMERICAL_FAILURE);
    }
  }
  return s;
}

static void prepare_numerics(libgto_pair &r) {
  auto &v = r.numerics;
  size_t P = r.primitives.size();
  int A = r.a.l, B = r.b.l;
  {
    bool quadratic = A + B == 2 && (A == 2 || B == 2);
    if ((A + B > 1 && !quadratic) || P == 0 || r.sensitive)
      return;
    v.low_degree = A + B;
    v.common_offset = r.primitives[0].offset;
    v.common_low = true;
    v.pmin = r.primitives[0].p;
    Real pmax = v.pmin;
    for (const auto &x : r.primitives) {
      v.common_low &= x.offset == v.common_offset;
      v.pmin = std::min(v.pmin, x.p);
      pmax = std::max(pmax, x.p);
    }
    if (!v.common_low && !quadratic)
      return;
    v.sqrt_pmax = std::sqrt(pmax);
    // Use already stable raw E0, never offset + a nearly cancelling center shift.
    if (A + B == 1)
      for (size_t p = 0; p < P; p++)
        for (int d = 0; d < 3; d++)
          v.active = std::max(v.active, std::abs(r.E[p * 12 + d]));
    size_t C = r.a.c * r.b.c;
    for (size_t c = 0; c < C; c++) {
      Real s1 = 0, s125 = 0, s15 = 0;
      for (size_t p = 0; p < P; p++) {
        const auto &x = r.primitives[p];
        Real t = std::abs(r.weights[p + P * c]) * x.K / x.p, root = std::sqrt(x.p);
        s1 += t;
        s125 += t / std::sqrt(root);
        s15 += t / root;
      }
      v.m1 = std::max(v.m1, s1);
      v.m125 = std::max(v.m125, s125);
      v.m15 = std::max(v.m15, s15);
    }
  }
}

void rounded_view(const std::vector<Real> &from, std::vector<double> &to, bool &safe) {
  to.resize(from.size());
  for (size_t i = 0; i < from.size(); i++) {
    if (std::abs(from[i]) > std::numeric_limits<double>::max()) {
      safe = false;
      to[i] = 0;
      continue;
    }
    to[i] = static_cast<double>(from[i]);
    if (from[i] != 0 && !std::isnormal(to[i]))
      safe = false;
  }
}

static void prepare_views(libgto_pair *r) {
  // Coefficient-independent rounded geometry and exact eligibility predicate.
  // The immutable owned pair lifetime is also the validity lifetime of this view.
  for (auto &v : r->primitives) {
    v.p64 = std::abs(v.p) > std::numeric_limits<double>::max()
                ? std::numeric_limits<double>::infinity()
                : double(v.p);
    v.K64 = std::abs(v.K) > std::numeric_limits<double>::max()
                ? std::numeric_limits<double>::infinity()
                : double(v.K);
    v.low_normal = std::isnormal(v.p64) && std::isnormal(v.K64);
  }

  const int A = r->a.l, B = r->b.l;
  const size_t P = r->primitives.size(), C = mul(r->a.c, r->b.c);
  rounded_view(r->E, r->E64, r->fp64_safe);
  rounded_view(r->pure_E, r->pure_E64, r->fp64_safe);
  rounded_view(r->weights, r->weights64, r->fp64_safe);

  auto masks = [](const std::vector<Real> &view, int n, std::vector<uint64_t> &out) {
    out.resize(view.size() / n);
    for (size_t row = 0; row < out.size(); row++) {
      uint64_t bits = 0;
      for (int c = 0; c < n; c++)
        if (view[row * n + c] != 0)
          bits |= uint64_t(1) << c;
      out[row] = bits;
    }
  };
  masks(r->E, r->N, r->E_mask);
  masks(r->pure_E, (2 * A + 1) * (2 * B + 1), r->pure_mask);
  r->sparse_cart = std::find(r->E_mask.begin(), r->E_mask.end(), uint64_t(0)) != r->E_mask.end();
  r->sparse_pure =
      std::find(r->pure_mask.begin(), r->pure_mask.end(), uint64_t(0)) != r->pure_mask.end();
  // Conservative column 1-norms of the rounded views; no duplicate tensor.
  auto prepare_norm = [&](const std::vector<double> &v, int n, std::vector<double> &result) {
    if (v.empty())
      return;
    result.resize(P * (n + 1));
    for (size_t p = 0; p < P; p++) {
      double largest = 0;
      for (int c = 0; c < n; c++) {
        Real sum = 0;
        for (int h = 0; h < r->H; h++)
          sum += std::abs(v[(p * r->H + h) * n + c]);
        double z =
            sum > std::numeric_limits<double>::max() /
                        (1 + 8 * std::numeric_limits<double>::epsilon())
                ? std::numeric_limits<double>::infinity()
                : static_cast<double>(sum) * (1 + 8 * std::numeric_limits<double>::epsilon()) +
                      std::numeric_limits<double>::min();
        result[p * (n + 1) + 1 + c] = z;
        largest = std::max(largest, z);
      }
      result[p * (n + 1)] = largest;
    }
  };
  prepare_norm(r->E64, r->N, r->norm_cart);
  prepare_norm(r->pure_E64, (2 * A + 1) * (2 * B + 1), r->norm_pure);
  Real mass = 0;
  for (size_t c = 0; c < C; c++) {
    Real sum = 0;
    for (size_t p = 0; p < P; p++)
      sum += std::abs(r->weights64[p + P * c]);
    mass = std::max(mass, sum);
  }
  r->weight_mass =
      mass > std::numeric_limits<double>::max() / (1 + 8 * std::numeric_limits<double>::epsilon())
          ? std::numeric_limits<double>::infinity()
          : static_cast<double>(mass) * (1 + 8 * std::numeric_limits<double>::epsilon());
  prepare_numerics(*r);
}

// A coefficient-dependent risk flag, not an integral-error certificate.
// For fixed homogeneous angular degree l, the radial overlap Gram matrix is
// proportional to (ai+aj)^(-l-3/2). The common positive angular factor cancels.
// Strong L2 cancellation can occur outside a narrow exponent-span condition.
static bool contraction_norm_risk(const Shell &s) {
  if (s.k < 2)
    return false;
  const Real amin = *std::min_element(s.a.begin(), s.a.end());
  std::vector<Real> v(s.k);
  auto kernel = [&](Real sum) {
    Real u = amin / sum;
    Real z = u * std::sqrt(u);
    for (int l = 0; l < s.l; l++)
      z *= u;
    return z;
  };
  for (size_t c = 0; c < s.c; c++) {
    bool positive = false, negative = false;
    Real largest = 0;
    for (size_t i = 0; i < s.k; i++) {
      Real w = s.weights[i + s.k * c];
      positive |= w > 0;
      negative |= w < 0;
      largest = std::max(largest, std::abs(w));
    }
    if (!positive || !negative)
      continue;
    for (size_t i = 0; i < s.k; i++)
      v[i] = s.weights[i + s.k * c] / largest;
    Real norm = 0, correction = 0, mass = 0;
    for (size_t i = 0; i < s.k; i++) {
      mass += std::abs(v[i]) * std::sqrt(kernel(2 * s.a[i]));
      for (size_t j = 0; j <= i; j++) {
        Real x = v[i] * v[j] * kernel(s.a[i] + s.a[j]);
        if (i != j)
          x *= 2;
        Real t = norm + x;
        correction += std::abs(norm) >= std::abs(x) ? (norm - t) + x : (x - t) + norm;
        norm = t;
      }
    }
    norm += correction;
    Real bound = mass * mass;
    // Four small shell norms can amplify cancellation across an ERI. This
    // conservative empirical cutoff routes risk to original-input MP evaluation.
    if (!std::isfinite(norm) || !std::isfinite(bound) || bound <= 0 || norm <= Real(1e-3L) * bound)
      return true;
  }
  return false;
}

std::unique_ptr<libgto_pair> make_pair(const libgto_shell &a, const libgto_shell &b) {
  auto r = std::make_unique<libgto_pair>();
  r->a = copy_shell(a);
  r->b = copy_shell(b);
  r->sensitive = contraction_norm_risk(r->a) || contraction_norm_risk(r->b);
  r->shape_sizes = {r->a.k, r->b.k, r->a.c, r->b.c};
  r->shape_bits = unsigned(r->a.l) | (unsigned(r->b.l) << 3) | (unsigned(r->a.norm) << 6) |
                  (unsigned(r->b.norm) << 7);
  r->shape_hash = r->shape_bits;
  for (size_t v : r->shape_sizes)
    r->shape_hash ^= v + size_t(0x9e3779b9) + (r->shape_hash << 6) + (r->shape_hash >> 2);
  // Near-coincident exponent rows with opposite signed radial coefficients can
  // lose far more than FP64 precision during contraction. This is a mathematical
  // feature guard, not a benchmark identity or rigorous condition estimate.
  for (const Shell *shell : {&r->a, &r->b})
    if (shell->k > 1) {
      auto mm = std::minmax_element(shell->a.begin(), shell->a.end());
      if (*mm.second - *mm.first <= 0.1L * (*mm.second))
        for (size_t c = 0; c < shell->c; c++) {
          Real sum = 0, mass = 0;
          for (size_t k = 0; k < shell->k; k++) {
            Real v = shell->weights[k + shell->k * c];
            sum += v;
            mass += std::abs(v);
          }
          if (mass > 0 && std::abs(sum) < 0.1L * mass)
            r->sensitive = true;
        }
    }
  if (r->a.l > 2 || r->b.l > 2) {
    prepare_high_pair(*r);
    return r;
  }
  int A = r->a.l, B = r->b.l, L = A + B;
  r->H = nh(L);
  r->N = ncart(A) * ncart(B);
  size_t P = mul(r->a.k, r->b.k), C = mul(r->a.c, r->b.c);
  if (A == 2 || B == 2)
    r->pure_E.resize(mul(mul(P, r->H), (2 * A + 1) * (2 * B + 1)));
  r->primitives.resize(P);
  r->E.resize(mul(mul(P, r->H), r->N));
  r->weights.resize(mul(P, C));
  auto oa = orbitals(A), ob = orbitals(B), hs = hermites(L);
  Real delta[3], distance2 = 0;
  for (int d = 0; d < 3; d++) {
    delta[d] = r->b.r[d] - r->a.r[d];
    distance2 += delta[d] * delta[d];
  }
  for (size_t j = 0; j < r->b.k; j++)
    for (size_t i = 0; i < r->a.k; i++) {
      size_t z = i + r->a.k * j;
      auto &v = r->primitives[z];
      Real alpha = r->a.a[i], beta = r->b.a[j];
      v.p = alpha + beta;
      require(std::isfinite(v.p), LIBGTO_NUMERICAL_FAILURE);
      for (int d = 0; d < 3; d++)
        v.offset[d] = beta / v.p * delta[d];
      v.K = std::exp(-(alpha / v.p) * beta * distance2);
      require(std::isfinite(v.K), LIBGTO_NUMERICAL_FAILURE);
      Real e[3][3][3][5] = {};
      for (int d = 0; d < 3; d++) {
        e[d][0][0][0] = 1.;
        for (int ia = 0; ia <= A; ia++)
          for (int ib = 0; ib <= B; ib++) {
            if (!ia && !ib)
              continue;
            int pa = ia ? ia - 1 : 0, pb = ia ? ib : ib - 1;
            Real shift = ia ? v.offset[d] : -(alpha / v.p) * delta[d];
            for (int t = 0; t <= ia + ib; t++)
              e[d][ia][ib][t] = (t ? e[d][pa][pb][t - 1] / (2 * v.p) : 0) +
                                shift * e[d][pa][pb][t] +
                                (t < 4 ? (t + 1) * e[d][pa][pb][t + 1] : 0);
          }
      }
      for (int h = 0; h < r->H; h++)
        for (size_t cb = 0; cb < ob.size(); cb++)
          for (size_t ca = 0; ca < oa.size(); ca++) {
            Real vE = 1;
            for (int d = 0; d < 3; d++)
              vE *= e[d][oa[ca][d]][ob[cb][d]][hs[h][d]];
            require(std::isfinite(vE), LIBGTO_NUMERICAL_FAILURE);
            r->E[(z * r->H + h) * r->N + ca + oa.size() * cb] = vE;
          }
      // The real-harmonic map is primitive-independent. Project the coefficient-
      // free E view once; channel weights and two-pair Coulomb R remain separate.
      auto transform = [](int l, int row, int col) -> Real {
        if (l == 0)
          return 1;
        if (l == 1)
          return col == row ? 1 : 0;
        const Real s = std::sqrt(3.L);
        if (row == 0)
          return col == 1 ? s : 0;
        if (row == 1)
          return col == 4 ? s : 0;
        if (row == 2)
          return col == 5 ? 1 : (col == 0 || col == 3 ? -.5L : 0);
        if (row == 3)
          return col == 2 ? s : 0;
        return col == 0 ? s / 2 : (col == 3 ? -s / 2 : 0);
      };
      if (A == 2 || B == 2) {
        int na = 2 * A + 1, nb = 2 * B + 1;
        for (int h = 0; h < r->H; h++)
          for (int b = 0; b < nb; b++)
            for (int a = 0; a < na; a++) {
              Real sum = 0;
              for (size_t j = 0; j < ob.size(); j++)
                for (size_t i = 0; i < oa.size(); i++)
                  sum += transform(A, a, i) * transform(B, b, j) *
                         r->E[(z * r->H + h) * r->N + i + oa.size() * j];
              r->pure_E[(z * r->H + h) * na * nb + a + na * b] = sum;
            }
      }
      for (size_t cb = 0; cb < r->b.c; cb++)
        for (size_t ca = 0; ca < r->a.c; ca++) {
          Real w = r->a.weights[i + r->a.k * ca] * r->b.weights[j + r->b.k * cb];
          require(std::isfinite(w), LIBGTO_NUMERICAL_FAILURE);
          r->weights[z + P * (ca + r->a.c * cb)] = w;
        }
    }
  prepare_views(r.get());

  // Optional exact dependency reduction. Keep the original arrays for REFERENCE.
  // At coincident centers, equal exponent sums imply identical pair geometry/E.
  // Coalesce only same-sign weights in every channel; avoid moving cancellation
  // into preparation. Machine-zero Kab terms already vanish in the reference.
  std::vector<std::vector<size_t>> groups;
  std::map<Real, size_t> lookup;
  std::vector<size_t> which(P, 0);
  bool same_center = r->a.r == r->b.r;
  for (size_t z = 0; z < P; z++) {
    if (r->primitives[z].K == 0) {
      which[z] = size_t(-1);
      continue;
    }
    auto it = same_center ? lookup.find(r->primitives[z].p) : lookup.end();
    if (it == lookup.end()) {
      which[z] = groups.size();
      if (same_center)
        lookup.emplace(r->primitives[z].p, groups.size());
      groups.push_back({z});
    } else {
      which[z] = it->second;
      groups[it->second].push_back(z);
    }
  }
  std::vector<bool> safe(groups.size(), true);
  for (size_t g = 0; g < groups.size(); g++) {
    size_t leader = groups[g][0];
    for (size_t z : groups[g]) {
      if (!std::equal(r->E.begin() + leader * r->H * r->N,
                      r->E.begin() + (leader + 1) * r->H * r->N, r->E.begin() + z * r->H * r->N))
        safe[g] = false;
      if (!r->pure_E.empty()) {
        size_t n = size_t(r->H) * (2 * A + 1) * (2 * B + 1);
        if (!std::equal(r->pure_E.begin() + leader * n, r->pure_E.begin() + (leader + 1) * n,
                        r->pure_E.begin() + z * n))
          safe[g] = false;
      }
    }
    for (size_t c = 0; c < C; c++) {
      bool positive = false, negative = false;
      for (size_t z : groups[g]) {
        Real w = r->weights[z + P * c];
        positive |= w > 0;
        negative |= w < 0;
      }
      if (positive && negative)
        safe[g] = false;
    }
  }
  std::vector<std::vector<size_t>> entries;
  std::vector<bool> done(groups.size());
  for (size_t z = 0; z < P; z++) {
    size_t g = which[z];
    if (g == size_t(-1))
      continue;
    if (safe[g]) {
      if (!done[g]) {
        entries.push_back(groups[g]);
        done[g] = true;
      }
    } else
      entries.push_back({z});
  }
  if (entries.size() < P) {
    auto child = std::make_unique<libgto_pair>();
    child->a = r->a;
    child->b = r->b;
    child->H = r->H;
    child->N = r->N;
    size_t n = entries.size(), cart_row = size_t(r->H) * r->N,
           pure_row = size_t(r->H) * (2 * A + 1) * (2 * B + 1);
    child->primitives.resize(n);
    child->E.resize(n * cart_row);
    if (!r->pure_E.empty())
      child->pure_E.resize(n * pure_row);
    child->weights.resize(n * C);
    for (size_t g = 0; g < n; g++) {
      size_t leader = entries[g][0];
      child->primitives[g] = r->primitives[leader];
      std::copy_n(r->E.data() + leader * cart_row, cart_row, child->E.data() + g * cart_row);
      if (!r->pure_E.empty())
        std::copy_n(r->pure_E.data() + leader * pure_row, pure_row,
                    child->pure_E.data() + g * pure_row);
      for (size_t c = 0; c < C; c++) {
        Real sum = 0, correction = 0;
        for (size_t z : entries[g]) {
          Real x = r->weights[z + P * c], t = sum + x;
          correction += std::abs(sum) >= std::abs(x) ? (sum - t) + x : (x - t) + sum;
          sum = t;
        }
        child->weights[g + n * c] = sum + correction;
        require(std::isfinite(sum + correction), LIBGTO_NUMERICAL_FAILURE);
      }
    }
    prepare_views(child.get());

    r->optimized = std::move(child);
    // Parent is the original uncompressed Cartesian/late-pure reference only.
    std::vector<Real>().swap(r->pure_E);
    std::vector<double>().swap(r->E64);
    std::vector<double>().swap(r->pure_E64);
    std::vector<double>().swap(r->weights64);
    std::vector<uint64_t>().swap(r->E_mask);
    std::vector<uint64_t>().swap(r->pure_mask);
    std::vector<double>().swap(r->norm_cart);
    std::vector<double>().swap(r->norm_pure);
  }
  return r;
}
}
