// Mixed-range amplitudes. Every original primitive is evaluated; no screening.

template <bool OneP>
static bool compute_scaled_low(const LowContext &ctx, size_t p, size_t q, Real *out, double budget,
                               bool active_ab = true) {
  constexpr int family = OneP ? 1 : 0;

  const auto &x = ctx.pa[p];
  const auto &y = ctx.pb[q];
  double ap = x.p64, bp = y.p64;
  if (!(ap >= 1e-8 && ap <= 1e8 && bp >= 1e-8 && bp <= 1e8) || !std::isnormal(x.K) ||
      !std::isnormal(y.K) || !(budget > 0))
    return false;
  double sum = ap + bp, rho = ap / sum * bp;
  if (!std::isnormal(sum) || !std::isnormal(rho) || std::min(ap, bp) / sum < 1e-12)
    return false;
  double radial = double(2 * std::pow(pi, 2.5L)) / (ap * bp * std::sqrt(sum));
  if (!std::isnormal(radial))
    return false;
  if constexpr (OneP) {
    double old = radial * x.K64 * y.K64;
    if (x.low_normal && y.low_normal && std::isnormal(old))
      return false;
  }
  double T = 0, X[3];
  for (int d = 0; d < 3; d++) {
    if (std::abs(ctx.delta[d]) + std::abs(x.offset[d]) + std::abs(y.offset[d]) > 1e8)
      return false;
    X[d] = double(ctx.delta[d] + (x.offset[d] - y.offset[d]));
    T += rho * X[d] * X[d];
  }
  if (!std::isfinite(T) || T < 0)
    return false;
  double F[2];
  if constexpr (OneP)
    low_boys_fixed<1>(T, F);
  else
    low_boys_fixed<0>(T, F);
  if (!std::isnormal(F[0]) || !(F[0] > 0))
    return false;

  Real pref = Real(radial) * x.K * y.K;
  if (!std::isnormal(pref))
    return false;
  Real value[3];
  double envelope = 1;
  if constexpr (OneP) {
    const double *e = (active_ab ? ctx.ea + 12 * p : ctx.ec + 12 * q);
    for (int d = 0; d < 3; d++) {
      double z = (active_ab ? -2 * rho : 2 * rho) * e[(3 - d) * 3 + d] * X[d];
      double g = e[d] * F[0] + z * F[1];
      if (!std::isfinite(g))
        return false;
      value[d] = pref * Real(g);
      envelope = std::max(envelope, std::abs(e[d]) + std::abs(z));
    }
  } else
    value[0] = pref * Real(F[0]);
  // Additional binary64 underflow allowance is evaluated before amplitude scaling.
  Real error = std::abs(pref) *
               (Real(F[0]) * Real(envelope) *
                    (4096 * std::numeric_limits<double>::epsilon() * (1 + std::min(T, 2.))) +
                Real(64) * std::numeric_limits<double>::denorm_min());
  error =
      error * (1 + 64 * std::numeric_limits<double>::epsilon()) + std::numeric_limits<Real>::min();
  if (!std::isfinite(error) || error > Real(budget))
    return false;
  for (int d = 0; d < (OneP ? 3 : 1); d++) {
    if (!std::isfinite(value[d]))
      return false;
    out[d] = value[d];
  }

  return true;
}
static bool compute_scaled_scalar(const LowContext &ctx, size_t p, size_t q, Real *out,
                                  double budget) {
  const auto &x = ctx.pa[p];
  const auto &y = ctx.pb[q];
  double ap = x.p64, bp = y.p64;
  if (!(ap >= 1e-8 && ap <= 1e8 && bp >= 1e-8 && bp <= 1e8) || !std::isnormal(x.K) ||
      !std::isnormal(y.K) || !(budget > 0))
    return false;
  double sum = ap + bp, rho = ap / sum * bp;
  if (!std::isnormal(sum) || !std::isnormal(rho) || std::min(ap, bp) / sum < 1e-12)
    return false;
  double T = 0;
  for (int d = 0; d < 3; d++) {
    if (std::abs(ctx.delta[d]) + std::abs(x.offset[d]) + std::abs(y.offset[d]) > 1e8)
      return false;
    double X = double(ctx.delta[d] + (x.offset[d] - y.offset[d]));
    T += rho * X * X;
  }
  if (!std::isfinite(T) || T < 0)
    return false;
  double F[1];
  low_boys_fixed<0>(T, F);
  if (!std::isnormal(F[0]) || !(F[0] > 0))
    return false;
  Real pref = 2 * std::pow(pi, 2.5L) / (x.p * y.p * std::sqrt(x.p + y.p)) * x.K * y.K;
  if (!std::isnormal(pref))
    return false;
  Real value = pref * Real(F[0]);
  if (!std::isnormal(value))
    return false;
  // The old budget and larger relative envelope are retained. This empirical
  // monitor is checked in wide arithmetic, before amplitude can underflow FP64.
  Real error = std::abs(value) *
               (Real(256) * std::numeric_limits<double>::epsilon() * (1 + std::min(T, .5))) *
               Real(ctx.an[p * 2]) * Real(ctx.bn[q * 2]);
  error =
      error * (1 + 64 * std::numeric_limits<double>::epsilon()) + std::numeric_limits<Real>::min();
  if (!std::isfinite(error) || error > Real(budget))
    return false;
  *out = value;
  return true;
}
