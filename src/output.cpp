#include "internal.hpp"
#include "pure_output.hpp"
namespace gto {
// Canonical Cartesian order used solely by this output convention adapter.
struct CartesianOrder {
  std::array<std::array<Triple, 6>, 3> cart{};
  constexpr CartesianOrder() {
    for (int l = 0; l <= 2; l++) {
      int i = 0;
      for (int x = l; x >= 0; x--)
        for (int y = l - x; y >= 0; y--)
          cart[l][i++] = {x, y, l - x - y};
    }
  }
};
static constexpr CartesianOrder output_order;
static bool store_transformed_output(const libgto_pair &a, const libgto_pair &b,
                                     const libgto_eri_options &o, libgto_workspace &w,
                                     const Real *contracted, Real *stage, double *out,
                                     const Layout &l) {
  double start = o.profile ? now() : 0;
  const Shell *ss[4] = {&a.a, &a.b, &b.a, &b.b};
  size_t nc[4], dim[4];
  Real scales[4][6];
  for (int s = 0; s < 4; ++s) {
    nc[s] = o.representation == LIBGTO_PURE ? 2 * ss[s]->l + 1 : ncart(ss[s]->l);
    dim[s] = nc[s] * ss[s]->c;
    for (size_t i = 0; i < nc[s]; ++i) {
      scales[s][i] = 1;
      if (ss[s]->norm == LIBGTO_NORMALIZED_PRIMITIVES && o.representation == LIBGTO_CARTESIAN) {
        Real df = 1;
        for (int x : output_order.cart[ss[s]->l][i])
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
              auto internal_index = [&](size_t component, int axis) {
                return o.representation == LIBGTO_PURE && ss[axis]->l == 1 ? (component + 1) % 3
                                                                           : component;
              };
              size_t v = internal_index(i, 0) +
                         nc[0] * (internal_index(j, 1) +
                                  nc[1] * (internal_index(c, 2) + nc[2] * internal_index(d, 3)));
              size_t dst =
                  (i + nc[0] * (ca % a.a.c)) +
                  dim[0] *
                      ((j + nc[1] * (ca / a.a.c)) +
                       dim[1] * ((c + nc[2] * (cb % b.a.c)) + dim[2] * (d + nc[3] * (cb / b.a.c))));
              stage[dst] = contracted[v + l.g * (ca + l.cab * cb)] * scales[0][i] * scales[1][j] *
                           scales[2][c] * scales[3][d];
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
  return true;
}

bool store_md_eri_output(const libgto_pair &a, const libgto_pair &b, const libgto_eri_options &o,
                         libgto_workspace &w, const Real *contracted, Real *stage, double *out,
                         const Layout &l) {
  if (o.representation != LIBGTO_PURE)
    return store_transformed_output(a, b, o, w, contracted, stage, out, l);
  double begin = o.profile ? now() : 0;
  store_pure_contracted(a, b, contracted, out, l);
  if (o.profile) {
    w.stats.transform_seconds = 0;
    w.stats.output_seconds = now() - begin;
  }
  return true;
}

}
