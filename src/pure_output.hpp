#pragma once
#include "internal.hpp"
#include <type_traits>
namespace gto {
template <class Scalar>
inline void store_pure_contracted(const libgto_pair &a, const libgto_pair &b,
                                  const Scalar *contracted, double *out, const Layout &l) {
  const Shell *ss[4] = {&a.a, &a.b, &b.a, &b.b};
  size_t nc[4], dim[4];
  unsigned char perm[4][9];
  bool identity = l.cab == 1 && l.ccd == 1;
  for (int s = 0; s < 4; s++) {
    nc[s] = 2 * ss[s]->l + 1;
    dim[s] = nc[s] * ss[s]->c;
    identity &= ss[s]->l != 1;
    for (size_t i = 0; i < nc[s]; i++)
      perm[s][i] = ss[s]->l == 1 ? (i + 1) % 3 : i;
  }
  auto write = [&](size_t to, const Scalar &v) {
    using std::abs;
    if constexpr (std::is_floating_point_v<Scalar>)
      require(std::isfinite(v), LIBGTO_NUMERICAL_FAILURE);
    require(v == v && abs(v) <= Scalar(std::numeric_limits<double>::max()),
            LIBGTO_NUMERICAL_FAILURE);
    out[to] = static_cast<double>(v);
  };
  if (identity) {
    for (size_t i = 0; i < l.public_size.elements; i++)
      write(i, contracted[i]);
    return;
  }
  for (size_t cb = 0; cb < l.ccd; cb++) {
    size_t c0 = cb % b.a.c, c1 = cb / b.a.c;
    for (size_t ca = 0; ca < l.cab; ca++) {
      size_t a0 = ca % a.a.c, a1 = ca / a.a.c,
             base =
                 nc[0] * a0 + dim[0] * (nc[1] * a1 + dim[1] * (nc[2] * c0 + dim[2] * nc[3] * c1));
      const Scalar *src = contracted + l.g * (ca + l.cab * cb);
      for (size_t d = 0; d < nc[3]; d++)
        for (size_t c = 0; c < nc[2]; c++)
          for (size_t j = 0; j < nc[1]; j++)
            for (size_t i = 0; i < nc[0]; i++) {
              size_t from =
                  perm[0][i] + nc[0] * (perm[1][j] + nc[1] * (perm[2][c] + nc[2] * perm[3][d]));
              write(base + i + dim[0] * (j + dim[1] * (c + dim[2] * d)), src[from]);
            }
    }
  }
}
}
