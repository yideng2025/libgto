#include "internal.hpp"
namespace gto {
static void boys_real(unsigned m, Real T, Real *f) {
  if (T < 32) {
    Real term = 1, sum = 1;
    unsigned k = 0;
    for (; k < 512; ++k) {
      term *= 2 * T / (2 * m + 2 * k + 3);
      sum += term;
      Real ratio = 2 * T / (2 * m + 2 * k + 5);
      // Bound the remaining positive tail after its ratio falls below 1.
      if (ratio < 1 && term * ratio / (1 - ratio) <= std::numeric_limits<Real>::epsilon() * sum)
        break;
    }
    require(k < 512, LIBGTO_NUMERICAL_FAILURE);
    Real e = std::exp(-T);
    f[m] = e * sum / (2 * m + 1);
    for (unsigned n = m; n > 0; --n)
      f[n - 1] = (2 * T * f[n] + e) / (2 * n - 1);
  } else {
    Real root = std::sqrt(T), e = std::exp(-T);
    f[0] = std::sqrt(pi) * std::erf(root) / (2 * root);
    for (unsigned n = 0; n < m; ++n)
      f[n + 1] = ((2 * n + 1) * f[n] - e) / (2 * T);
  }
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

static void primitive(const libgto_pair &a, const libgto_pair &b, size_t p, size_t q,
                      const Layout &l, Real *R, Real *M, Real *U, Real *G) {
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
  int offset[9] = {};
  for (int m = 1; m <= l.L; ++m)
    offset[m] = offset[m - 1] + nh(l.L - m + 1);
  Real factor = 1;
  for (int m = 0; m <= l.L; ++m) {
    R[offset[m]] = factor * F[m];
    factor *= -2 * rho;
  }
  // Only m + t + u + v <= L is stored; each entry is evaluated once.
  for (int m = l.L - 1; m >= 0; --m) {
    for (int h = 1; h < nh(l.L - m); ++h) {
      auto t = angular.h[h];
      int axis = t[0] ? 0 : (t[1] ? 1 : 2), degree = t[axis];
      --t[axis];
      Real value = X[axis] * R[offset[m + 1] + hi(t[0], t[1], t[2])];
      if (degree > 1) {
        --t[axis];
        value += (degree - 1) * R[offset[m + 1] + hi(t[0], t[1], t[2])];
      }
      R[offset[m] + h] = value;
    }
  }
  Real prefactor = 2 * std::pow(pi, 2.5L) / (x.p * y.p * std::sqrt(x.p + y.p)) * x.K * y.K;
  for (int h = 0; h < l.hab; ++h) {
    for (int k = 0; k < l.hcd; ++k) {
      auto s = angular.h[h], t = angular.h[k];
      Real phase = (t[0] + t[1] + t[2]) % 2 ? -1 : 1;
      M[h * l.hcd + k] = phase * prefactor * R[hi(s[0] + t[0], s[1] + t[1], s[2] + t[2])];
    }
  }
  const Real *ea = a.E.data() + p * l.hab * l.nab;
  const Real *ec = b.E.data() + q * l.hcd * l.ncd;
  std::fill_n(U, l.u, 0);
  std::fill_n(G, l.g, 0);
  // U = M Ecd; G = Eab^T U. Rightmost AO indices are contiguous.
  for (int h = 0; h < l.hab; ++h) {
    for (int k = 0; k < l.hcd; ++k) {
      Real value = M[h * l.hcd + k];
      for (int cd = 0; cd < l.ncd; ++cd)
        U[h * l.ncd + cd] += value * ec[k * l.ncd + cd];
    }
  }
  for (int h = 0; h < l.hab; ++h) {
    for (int cd = 0; cd < l.ncd; ++cd) {
      Real value = U[h * l.ncd + cd];
      for (int ab = 0; ab < l.nab; ++ab)
        G[ab + l.nab * cd] += ea[h * l.nab + ab] * value;
    }
  }
}

static Real transform(int l, int row, int column) {
  if (!l)
    return 1;
  if (l == 1) {
    static const int map[3] = {1, 2, 0}; // y,z,x from Cartesian x,y,z.
    return column == map[row] ? 1 : 0;
  }
  const Real s = std::sqrt(3.L);
  switch (row) {
  case 0:
    return column == 1 ? s : 0;
  case 1:
    return column == 4 ? s : 0;
  case 2:
    return column == 5 ? 1 : (column == 0 || column == 3 ? -.5L : 0);
  case 3:
    return column == 2 ? s : 0;
  default:
    return column == 0 ? s * .5L : (column == 3 ? -s * .5L : 0);
  }
}

void compute_reference(const libgto_pair &a, const libgto_pair &b, const libgto_eri_options &o,
                       libgto_workspace &w, double *out, const Layout &l) {
  w.stats = {};
  Real *R = w.arena.data();
  Real *M = R + l.r;
  Real *U = M + l.m;
  Real *G = U + l.u;
  Real *tmp = G + l.g;
  Real *contracted = tmp + l.tmp;
  Real *stage = contracted + l.contracted;
  Real *other = stage + l.stage;
  std::fill_n(contracted, l.contracted, 0);
  size_t P = a.primitives.size(), Q = b.primitives.size();
  auto evaluate = [&](size_t p, size_t q) {
    double t = o.profile ? now() : 0;
    primitive(a, b, p, q, l, R, M, U, G);
    if (o.profile)
      w.stats.recurrence_seconds += now() - t;
    ++w.stats.primitive_quartets;
  };
  if (l.cab == 1 && l.ccd == 1) {
    for (size_t q = 0; q < Q; ++q) {
      for (size_t p = 0; p < P; ++p) {
        evaluate(p, q);
        double t = o.profile ? now() : 0;
        Real weight = a.weights[p] * b.weights[q];
        for (size_t v = 0; v < l.g; ++v)
          contracted[v] += weight * G[v];
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
        evaluate(l.abfirst ? p : q, l.abfirst ? q : p);
        double t = o.profile ? now() : 0;
        for (size_t c = 0; c < ninner; ++c) {
          Real weight = wi[p + inner * c];
          for (size_t v = 0; v < l.g; ++v)
            tmp[v + l.g * c] += weight * G[v];
        }
        if (o.profile)
          w.stats.contraction_seconds += now() - t;
      }
      double t = o.profile ? now() : 0;
      for (size_t d = 0; d < nouter; ++d) {
        for (size_t c = 0; c < ninner; ++c) {
          Real weight = wo[q + outer * d];
          size_t slot = l.abfirst ? c + l.cab * d : d + l.cab * c;
          for (size_t v = 0; v < l.g; ++v)
            contracted[v + l.g * slot] += weight * tmp[v + l.g * c];
        }
      }
      if (o.profile)
        w.stats.contraction_seconds += now() - t;
    }
  }
  double start = o.profile ? now() : 0;
  const Shell *ss[4] = {&a.a, &a.b, &b.a, &b.b};
  size_t nc[4], dim[4];
  Real scales[4][6];
  for (int s = 0; s < 4; ++s) {
    nc[s] = ncart(ss[s]->l);
    dim[s] = nc[s] * ss[s]->c;
    for (size_t i = 0; i < nc[s]; ++i) {
      scales[s][i] = 1;
      if (ss[s]->norm == LIBGTO_NORMALIZED_PRIMITIVES && o.representation == LIBGTO_CARTESIAN) {
        Real df = 1;
        for (int x : angular.cart[ss[s]->l][i])
          if (x == 2)
            df *= 3;
        scales[s][i] = std::sqrt((ss[s]->l == 2 ? 3.L : 1.L) / df);
      }
    }
  }
  // Fold contraction columns into each axis of the complete canonical block.
  for (size_t cb = 0; cb < l.ccd; ++cb)
    for (size_t ca = 0; ca < l.cab; ++ca)
      for (size_t d = 0; d < nc[3]; ++d)
        for (size_t c = 0; c < nc[2]; ++c)
          for (size_t j = 0; j < nc[1]; ++j)
            for (size_t i = 0; i < nc[0]; ++i) {
              size_t v = i + nc[0] * (j + nc[1] * (c + nc[2] * d));
              size_t dst =
                  (i + nc[0] * (ca % a.a.c)) +
                  dim[0] *
                      ((j + nc[1] * (ca / a.a.c)) +
                       dim[1] * ((c + nc[2] * (cb % b.a.c)) + dim[2] * (d + nc[3] * (cb / b.a.c))));
              stage[dst] = contracted[v + l.g * (ca + l.cab * cb)] * scales[0][i] * scales[1][j] *
                           scales[2][c] * scales[3][d];
            }
  if (o.representation == LIBGTO_PURE) {
    size_t total = l.contracted;
    for (int axis = 0; axis < 4; ++axis) {
      size_t stride = 1;
      for (int s = 0; s < axis; ++s)
        stride *= dim[s];
      size_t olddim = dim[axis], pure = 2 * ss[axis]->l + 1;
      size_t newdim = pure * ss[axis]->c, outer = total / (stride * olddim);
      for (size_t block = 0; block < outer; ++block)
        for (size_t ch = 0; ch < ss[axis]->c; ++ch)
          for (size_t row = 0; row < pure; ++row)
            for (size_t k = 0; k < stride; ++k) {
              Real value = 0;
              for (size_t col = 0; col < nc[axis]; ++col)
                value += transform(ss[axis]->l, row, col) *
                         stage[k + stride * (col + nc[axis] * ch + olddim * block)];
              other[k + stride * (row + pure * ch + newdim * block)] = value;
            }
      total = total / olddim * newdim;
      dim[axis] = newdim;
      std::swap(stage, other);
    }
  }
  if (o.profile)
    w.stats.transform_seconds = now() - start;
  start = o.profile ? now() : 0;
  for (size_t i = 0; i < l.public_size.elements; ++i) {
    require(std::isfinite(stage[i]) && std::abs(stage[i]) <= std::numeric_limits<double>::max(),
            LIBGTO_NUMERICAL_FAILURE);
    out[i] = static_cast<double>(stage[i]);
  }
  if (o.profile)
    w.stats.output_seconds = now() - start;
}
}
