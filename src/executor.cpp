#include "internal.hpp"
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace gto {
static void execute_legacy(libgto_batch *b, const libgto_eri_task *t, size_t n) {
  gto::require(b && (!n || t));
  if (!n)
    return;
  // Validate all descriptors before writing any output. Tiles are claimed lazily;
  // only their small [begin,end) range lives in each worker.
  size_t maxarena = 0, maxview = 0, maxfast = 0, mh = 0;
  bool precise = false;
  for (size_t i = 0; i < n; i++) {
    gto::require(t[i].ab && t[i].cd && t[i].output);
    auto l = gto::layout(*t[i].ab, *t[i].cd, t[i].options);
    gto::require(l.public_size.workspace_bytes <= b->options.max_workspace_bytes,
                 LIBGTO_RESOURCE_LIMIT);
    gto::require(t[i].output_capacity >= l.public_size.elements, LIBGTO_BUFFER_TOO_SMALL);
    maxarena = std::max(maxarena, l.total);
    maxview = std::max(maxview, l.borrowed);
    maxfast = std::max(maxfast, l.fast);
    mh = std::max(mh, high_workspace_extra(l));
    precise |= needs_sensitive(*t[i].ab, *t[i].cd, l, b->work.front().policy);
  }
  size_t extra = add(precise ? sensitive_workspace_bytes() : 0, mh);
  gto::require(gto::add(gto::add(gto::mul(maxarena, sizeof(gto::Real)),
                                 gto::mul(gto::add(maxview, maxfast), sizeof(double))),
                        extra) <= b->options.max_workspace_bytes,
               LIBGTO_RESOURCE_LIMIT);
  for (auto &w : b->work)
    if (gto::add(gto::add(gto::mul(std::max(maxarena, w.arena.capacity()), sizeof(gto::Real)),
                          gto::mul(gto::add(std::max(maxview, w.borrowed.capacity()),
                                            std::max(maxfast, w.fast.capacity())),
                                   sizeof(double))),
                 add(w.precise ? sensitive_workspace_bytes()
                               : (precise ? sensitive_workspace_bytes() : 0),
                     mh)) > b->options.max_workspace_bytes) {
      std::vector<gto::Real>().swap(w.arena);
      std::vector<double>().swap(w.borrowed);
      std::vector<double>().swap(w.fast);
      if (!precise)
        w.precise.reset();
    }
  // Respect the caller task cap and independent output/workspace limits.
  // Large complete blocks run alone.
  size_t tile_tasks = b->options.max_tile_tasks;
  std::atomic<size_t> next{0};
  std::atomic<int> status{0};
#ifdef _OPENMP
#pragma omp parallel num_threads(b->options.threads) if (n > 1)
#endif
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    while (!status.load(std::memory_order_relaxed)) {
      size_t begin = next.load(), end = begin;
      do {
        end = begin;
        if (begin >= n)
          break;
        size_t bytes = 0;
        while (end < n && end - begin < tile_tasks) {
          const auto &job = t[end];
          bool pure = job.options.representation == LIBGTO_PURE;
          // Full descriptors and overflow were checked before the parallel region.
          size_t size = sizeof(double) *
                        size_t(pure ? (2 * job.ab->a.l + 1) * (2 * job.ab->b.l + 1) : job.ab->N) *
                        size_t(pure ? (2 * job.cd->a.l + 1) * (2 * job.cd->b.l + 1) : job.cd->N) *
                        job.ab->a.c * job.ab->b.c * job.cd->a.c * job.cd->b.c;
          if (end > begin && size > b->options.max_tile_output_bytes - bytes)
            break;
          bytes += size;
          ++end;
          if (bytes >= b->options.max_tile_output_bytes)
            break;
        }
      } while (!next.compare_exchange_weak(begin, end));
      if (begin == end)
        break;
      for (size_t i = begin; i < end; i++) {
        auto s = libgto_eri(t[i].ab, t[i].cd, &t[i].options, &b->work[tid], t[i].output,
                            t[i].output_capacity);
        if (s != LIBGTO_SUCCESS) {
          status.store(int(s));
          break;
        }
      }
    }
  }
  gto::require(!status.load(), static_cast<libgto_status>(status.load()));
}

// allocation requirements pass a collective barrier before any output write.
// Each copied layout is shape-only; no cached plan or pair data is mutated.
static bool execute_parallel_preflight(libgto_batch *b, const libgto_eri_task *t, size_t n) {
#ifndef _OPENMP
  (void)b;
  (void)t;
  (void)n;
  return false;
#else
  if (n < 512 || b->options.threads < 8)
    return false;
  size_t need = add(add(sizeof(b->plans), sizeof(b->plan_index)),
                    add(mul(b->bindings.capacity(), sizeof(BlockBinding)),
                        add(mul(std::max(n, b->ranges.capacity()), sizeof(OutputRange)),
                            mul(std::max(n, b->parallel_layouts.capacity()), sizeof(Layout)))));
  if (need > b->binding_byte_limit) {
    std::vector<Layout>().swap(b->parallel_layouts);
    return false;
  }
  if (b->parallel_layouts.capacity() < n) {
    std::vector<Layout>().swap(b->parallel_layouts);
    b->parallel_layouts = std::vector<Layout>(n);
  } else if (b->parallel_layouts.size() < n)
    b->parallel_layouts.resize(n);
  if (b->ranges.capacity() < n) {
    std::vector<OutputRange>().swap(b->ranges);
    b->ranges = std::vector<OutputRange>(n);
  } else if (b->ranges.size() < n)
    b->ranges.resize(n);
  require(libgto_batch_binding_bytes(b) <= b->binding_byte_limit, LIBGTO_RESOURCE_LIMIT);
  size_t ma = 0, mv = 0, mf = 0, mh = 0, max_output = 0, builds = 0, first_bad = n;
  int precise = 0;
  std::atomic<int> status{0};
  std::atomic<size_t> next{0};
  size_t fixed_count = std::min(n, b->options.max_tile_tasks);
  bool fixed_claim = false;
#pragma omp parallel num_threads(b->options.threads)
  {
    int tid = omp_get_thread_num();
#pragma omp for schedule(static) reduction(max:ma,mv,mf,mh,max_output) reduction(+:builds) reduction(|:precise)
    for (size_t i = 0; i < n; i++) {
      auto result = protect([&] {
        const auto &task = t[i];
        require(task.ab && task.cd && task.output);
        auto plan = resolve_plan(eri_request(task.options, b->work.front().policy),
                                 quartet_shape(*task.ab, *task.cd));
        ++builds;
        auto &l = b->parallel_layouts[i];
        l = plan.storage;
        require(l.public_size.workspace_bytes <= b->options.max_workspace_bytes,
                LIBGTO_RESOURCE_LIMIT);
        require(task.output_capacity >= l.public_size.elements, LIBGTO_BUFFER_TOO_SMALL);
        auto address = reinterpret_cast<uintptr_t>(task.output);
        require(l.public_size.output_bytes <= std::numeric_limits<uintptr_t>::max() - address);
        b->ranges[i] = {address, address + l.public_size.output_bytes};
        ma = std::max(ma, l.total);
        mv = std::max(mv, l.borrowed);
        mf = std::max(mf, l.fast);
        mh = std::max(mh, high_workspace_extra(l));
        max_output = std::max(max_output, l.public_size.output_bytes);
        precise |= needs_sensitive(*task.ab, *task.cd, l, b->work.front().policy);
      });
      if (result != LIBGTO_SUCCESS) {
#pragma omp critical(libgto_preflight_error)
        {
          if (i < first_bad) {
            first_bad = i;
            status.store(int(result), std::memory_order_relaxed);
          }
        }
      }
    }
#pragma omp single
    {
      b->binding_stats.preflight_tasks += n;
      b->binding_stats.plan_builds += builds;
      b->binding_stats.bindings += n;
      if (!status.load(std::memory_order_relaxed)) {
        auto result = protect([&] {
          bool ordered = true;
          for (size_t i = 1; i < n; i++)
            if (b->ranges[i - 1].end > b->ranges[i].begin) {
              ordered = false;
              break;
            }
          if (!ordered) {
            std::sort(b->ranges.begin(), b->ranges.begin() + n,
                      [](const OutputRange &a, const OutputRange &b) { return a.begin < b.begin; });
            for (size_t i = 1; i < n; i++)
              require(b->ranges[i - 1].end <= b->ranges[i].begin);
          }
          size_t extra = add(precise ? sensitive_workspace_bytes() : 0, mh);
          require(add(add(mul(ma, sizeof(Real)), mul(add(mv, mf), sizeof(double))), extra) <=
                      b->options.max_workspace_bytes,
                  LIBGTO_RESOURCE_LIMIT);
          fixed_claim =
              max_output && fixed_count <= b->options.max_tile_output_bytes / max_output &&
              fixed_count <= (std::numeric_limits<size_t>::max() - n) / (b->work.size() + 1);
        });
        if (result != LIBGTO_SUCCESS)
          status.store(int(result), std::memory_order_relaxed);
      }
    }
    if (!status.load(std::memory_order_relaxed)) {
      auto result = protect([&] {
        auto &w = b->work[tid];
        size_t extra = add(precise ? sensitive_workspace_bytes() : 0, mh);
        if (add(add(mul(std::max(ma, w.arena.capacity()), sizeof(Real)),
                    mul(add(std::max(mv, w.borrowed.capacity()), std::max(mf, w.fast.capacity())),
                        sizeof(double))),
                add(w.precise ? sensitive_workspace_bytes()
                              : (precise ? sensitive_workspace_bytes() : 0),
                    mh)) > b->options.max_workspace_bytes) {
          std::vector<Real>().swap(w.arena);
          std::vector<double>().swap(w.borrowed);
          std::vector<double>().swap(w.fast);
          if (!precise)
            w.precise.reset();
        }
        reserve(w, ma, mv, mf, precise);
      });
      if (result != LIBGTO_SUCCESS)
        status.store(int(result), std::memory_order_relaxed);
    }
#pragma omp barrier
    while (!status.load(std::memory_order_relaxed)) {
      size_t begin, end;
      if (fixed_claim) {
        begin = next.fetch_add(fixed_count, std::memory_order_relaxed);
        if (begin >= n)
          break;
        end = begin + std::min(fixed_count, n - begin);
      } else {
        begin = next.load();
        end = begin;
        do {
          end = begin;
          if (begin >= n)
            break;
          size_t bytes = 0;
          while (end < n && end - begin < b->options.max_tile_tasks) {
            size_t size = b->parallel_layouts[end].public_size.output_bytes;
            if (end > begin && size > b->options.max_tile_output_bytes - bytes)
              break;
            bytes += size;
            ++end;
            if (bytes >= b->options.max_tile_output_bytes)
              break;
          }
        } while (!next.compare_exchange_weak(begin, end));
      }
      if (begin == end)
        break;
      auto result = protect([&] {
        for (size_t i = begin; i < end; i++) {
          const auto &job = t[i];
          compute(*job.ab, *job.cd, job.options, b->work[tid], job.output, b->parallel_layouts[i]);
        }
      });
      if (result != LIBGTO_SUCCESS)
        status.store(int(result), std::memory_order_relaxed);
    }
  }
  require(!status.load(), static_cast<libgto_status>(status.load()));
  return true;
#endif
}
void execute_batch(libgto_batch *b, const libgto_eri_task *t, size_t n) {
  require(b && (!n || t));
  if (!n)
    return;
  if (n > b->binding_limit) {
    ++b->binding_stats.legacy_batches;
    b->binding_stats.scalar_reentries += n;
    execute_legacy(b, t, n);
    return;
  }
  if (execute_parallel_preflight(b, t, n))
    return;
  struct Reset {
    libgto_batch &b;
    ~Reset() {
      b.active_bindings = 0;
    }
  } reset{*b};
  b->plan_count = 0;
  b->plan_index.fill(0);
  b->active_bindings = 0; // High-water storage is reused; only this call can bind it.
  size_t need = add(add(sizeof(b->plans), sizeof(b->plan_index)),
                    mul(n, sizeof(BlockBinding) + sizeof(OutputRange)));
  if (add(need, mul(b->parallel_layouts.capacity(), sizeof(Layout))) > b->binding_byte_limit)
    std::vector<Layout>().swap(b->parallel_layouts);
  require(need <= b->binding_byte_limit, LIBGTO_RESOURCE_LIMIT);
  // Reserve exact metadata capacity; old and new buffers never coexist on growth.
  if (b->bindings.capacity() < n || b->ranges.capacity() < n) {
    std::vector<BlockBinding>().swap(b->bindings);
    std::vector<OutputRange>().swap(b->ranges);
    b->bindings = std::vector<BlockBinding>(n);
    b->ranges = std::vector<OutputRange>(n);
  } else if (b->bindings.size() < n) {
    b->bindings.resize(n);
    b->ranges.resize(n);
  }
  require(libgto_batch_binding_bytes(b) <= b->binding_byte_limit, LIBGTO_RESOURCE_LIMIT);
  size_t ma = 0, mv = 0, mf = 0, mh = 0, max_output = 0;
  bool ordered = true, legacy = false, precise = false;
  for (size_t i = 0; i < n; i++) {
    const auto &task = t[i];
    require(task.ab && task.cd && task.output);
    ++b->binding_stats.preflight_tasks;
    const auto *plan = legacy ? nullptr : cached_plan(*b, task, b->work.front().policy);
    Layout uncached{};
    if (!plan) {
      legacy = true;
      uncached = layout(*task.ab, *task.cd, task.options);
    }
    const auto &l = plan ? plan->storage : uncached;
    require(l.public_size.workspace_bytes <= b->options.max_workspace_bytes, LIBGTO_RESOURCE_LIMIT);
    max_output = std::max(max_output, l.public_size.output_bytes);
    require(task.output_capacity >= l.public_size.elements, LIBGTO_BUFFER_TOO_SMALL);
    auto address = reinterpret_cast<uintptr_t>(task.output);
    require(l.public_size.output_bytes <= std::numeric_limits<uintptr_t>::max() - address);
    b->ranges[i] = {address, address + l.public_size.output_bytes};
    if (i && b->ranges[i - 1].end > address)
      ordered = false;
    b->bindings[i] = {&task, plan};
    b->active_bindings = i + 1;
    ++b->binding_stats.bindings;
    ma = std::max(ma, l.total);
    mv = std::max(mv, l.borrowed);
    mf = std::max(mf, l.fast);
    mh = std::max(mh, high_workspace_extra(l));
    precise |= needs_sensitive(*task.ab, *task.cd, l, b->work.front().policy);
  }
  if (!ordered) {
    std::sort(b->ranges.begin(), b->ranges.begin() + n,
              [](const OutputRange &a, const OutputRange &b) { return a.begin < b.begin; });
    for (size_t i = 1; i < n; i++)
      require(b->ranges[i - 1].end <= b->ranges[i].begin);
  }
  size_t extra = add(precise ? sensitive_workspace_bytes() : 0, mh);
  require(add(add(mul(ma, sizeof(Real)), mul(add(mv, mf), sizeof(double))), extra) <=
              b->options.max_workspace_bytes,
          LIBGTO_RESOURCE_LIMIT);
  if (legacy) {
    ++b->binding_stats.legacy_batches;
    b->binding_stats.scalar_reentries += n;
    execute_legacy(b, t, n);
    return;
  }
  bool grow = false;
  for (auto &w : b->work) {
    if (add(add(mul(std::max(ma, w.arena.capacity()), sizeof(Real)),
                mul(add(std::max(mv, w.borrowed.capacity()), std::max(mf, w.fast.capacity())),
                    sizeof(double))),
            add(w.precise ? sensitive_workspace_bytes()
                          : (precise ? sensitive_workspace_bytes() : 0),
                mh)) > b->options.max_workspace_bytes) {
      std::vector<Real>().swap(w.arena);
      std::vector<double>().swap(w.borrowed);
      std::vector<double>().swap(w.fast);
      if (!precise)
        w.precise.reset();
    }
    if (n > 1 || &w == &b->work.front())
      grow |= w.arena.size() < ma || w.fast.size() < mf || w.borrowed.size() < mv ||
              (precise && !w.precise);
  }
  const size_t fixed_count = std::min(n, b->options.max_tile_tasks);
  const bool fixed_claim =
      max_output && fixed_count <= b->options.max_tile_output_bytes / max_output &&
      fixed_count <= (std::numeric_limits<size_t>::max() - n) / (b->work.size() + 1);
  std::atomic<size_t> next{0};
  std::atomic<int> status{0};
#ifdef _OPENMP
#pragma omp parallel num_threads(b->options.threads) if (n > 1)
#endif
  {
    int tid = 0;
#ifdef _OPENMP
    tid = omp_get_thread_num();
#endif
    if (grow) {
      auto allocation = protect([&] { reserve(b->work[tid], ma, mv, mf, precise); });
      if (allocation != LIBGTO_SUCCESS)
        status.store(int(allocation));
#ifdef _OPENMP
#pragma omp barrier
#endif
    }
    while (!status.load(std::memory_order_relaxed)) {
      size_t begin, end;
      if (fixed_claim) {
        begin = next.fetch_add(fixed_count, std::memory_order_relaxed);
        if (begin >= n)
          break;
        end = begin + std::min(fixed_count, n - begin);
      } else {
        begin = next.load();
        end = begin;
        do {
          end = begin;
          if (begin >= n)
            break;
          size_t bytes = 0;
          while (end < n && end - begin < b->options.max_tile_tasks) {
            size_t size = b->bindings[end].plan->storage.public_size.output_bytes;
            if (end > begin && size > b->options.max_tile_output_bytes - bytes)
              break;
            bytes += size;
            ++end;
            if (bytes >= b->options.max_tile_output_bytes)
              break;
          }
        } while (!next.compare_exchange_weak(begin, end));
      }
      if (begin == end)
        break;
      auto result = protect([&] {
        for (size_t i = begin; i < end; i++) {
          const auto &binding = b->bindings[i];
          const auto &job = *binding.task;
          compute(*job.ab, *job.cd, job.options, b->work[tid], job.output, binding.plan->storage);
        }
      });
      if (result != LIBGTO_SUCCESS)
        status.store(int(result));
    }
  }
  // Descriptors/pairs/outputs are borrowed only during this synchronous call.
  // No binding is consulted after return; next call overwrites every active slot.
  require(!status.load(), static_cast<libgto_status>(status.load()));
}

}
