#include "internal.hpp"
namespace gto {
void reserve(libgto_workspace &w, size_t n, size_t view, size_t fast, bool precise) {
  size_t base =
      add(mul(std::max(n, w.arena.capacity()), sizeof(Real)),
          mul(add(std::max(view, w.borrowed.capacity()), std::max(fast, w.fast.capacity())),
              sizeof(double)));
  base = add(base, (precise || w.precise) ? sensitive_workspace_bytes() : 0);
  if (add(base, high_scratch_bytes(w)) > w.capacity_limit)
    w.high.reset();
  require(base <= w.capacity_limit, LIBGTO_RESOURCE_LIMIT);
  if (w.fast.size() < fast) {
    std::vector<double>().swap(w.fast);
    w.fast = std::vector<double>(fast);
  }
  if (w.arena.size() < n) {
    std::vector<Real>().swap(w.arena);
    w.arena = std::vector<Real>(n);
  }
  if (w.borrowed.size() < view) {
    std::vector<double>().swap(w.borrowed);
    w.borrowed = std::vector<double>(view);
  }
  if (precise)
    reserve_sensitive(w);
}
}
