static void pack_two_p(two_p::Packet &p, size_t k, const libgto_pair &a, const libgto_pair &b,
                       size_t ip, size_t iq, double budget) {
  const auto &x = a.primitives[ip];
  const auto &y = b.primitives[iq];
  p.valid[k] = x.low_normal && y.low_normal;
  double ap = x.p64, bp = y.p64, sum = ap + bp;
  p.rho[k] = ap / sum * bp;
  p.T[k] = 0;
  for (int d = 0; d < 3; d++) {
    p.X[d][k] = double((a.a.r[d] - b.a.r[d]) + (x.offset[d] - y.offset[d]));
    p.T[k] += p.rho[k] * p.X[d][k] * p.X[d][k];
    p.a0[d][k] = a.E64[ip * 12 + d];
    p.a1[d][k] = a.E64[ip * 12 + (3 - d) * 3 + d];
    p.c0[d][k] = b.E64[iq * 12 + d];
    p.c1[d][k] = b.E64[iq * 12 + (3 - d) * 3 + d];
  }
  p.pref[k] = double(2 * std::pow(pi, 2.5L)) / (ap * bp * std::sqrt(sum)) * x.K64 * y.K64;
  p.valid[k] &= std::isnormal(sum) && std::isnormal(p.rho[k]) && std::isnormal(p.pref[k]) &&
                std::isfinite(p.T[k]) && p.T[k] >= 0;
  p.na[k] = a.norm_cart[ip * (a.N + 1)];
  p.nb[k] = b.norm_cart[iq * (b.N + 1)];
  p.budget[k] = budget;
}
