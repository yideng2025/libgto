#ifndef LIBGTO_H
#define LIBGTO_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* The library evaluates four-center Coulomb ERIs only; no screening is applied.
   Inputs and returned values must use atomic units. Concurrent calls require
   distinct workspaces or batch handles. The API is under active development. */
/* C-compatible integer enums. A fixed C++ underlying type lets the C boundary
   validate out-of-range integer inputs without first invoking C++ enum UB. */
#ifdef __cplusplus
#define LIBGTO_ENUM enum : int
#else
#define LIBGTO_ENUM enum
#endif
/* Atomic units; Cartesian and real solid-harmonic shells through g (l <= 4). No implicit
   contraction normalization. Double input/output. AUTO selects monitored FP64, long-double, or
   local fixed-precision reconstruction according to numerical features and shape. The f/g path
   retains wide auxiliaries/weights, monitored FP64 angular products, and raw-input 512-bit
   reconstruction. Pair preparation and selected auxiliary stages retain long double. REFERENCE
   retains the original long-double path; it is not an exact oracle for arbitrary
   cancellation. Effective long-double precision is platform dependent. */
typedef LIBGTO_ENUM{LIBGTO_SUCCESS = 0,          LIBGTO_INVALID_ARGUMENT = 1,
                    LIBGTO_UNSUPPORTED = 2,      LIBGTO_ALLOCATION_FAILED = 3,
                    LIBGTO_BUFFER_TOO_SMALL = 4, LIBGTO_NUMERICAL_FAILURE = 5,
                    LIBGTO_RESOURCE_LIMIT = 6} libgto_status;
typedef LIBGTO_ENUM{LIBGTO_CARTESIAN = 0, LIBGTO_PURE = 1} libgto_representation;
typedef LIBGTO_ENUM{LIBGTO_UNNORMALIZED = 0, LIBGTO_NORMALIZED_PRIMITIVES = 1} libgto_normalization;
typedef LIBGTO_ENUM{LIBGTO_CONTRACT_AUTO = 0, LIBGTO_AB_FIRST = 1,
                    LIBGTO_CD_FIRST = 2} libgto_contraction_order;
typedef struct {
  int angular_momentum;
  double center[3]; /* Bohr */
  size_t primitive_count, channel_count;
  const double *exponents;    /* positive finite; primitive_count entries */
  const double *coefficients; /* primitive + primitive_count*channel */
  libgto_normalization normalization;
} libgto_shell;
/* UNNORMALIZED: coefficients multiply monomials (Cartesian) or regular real solid
   harmonics (pure). NORMALIZED_PRIMITIVES: each primitive component has unit L2
   norm in the requested representation; contractions are never renormalized.
   Cartesian: x descending then y descending, d=xx,xy,xz,yy,yz,zz.
   Pure: m=-l..l without the Condon--Shortley phase, p=y,z,x; d=sqrt(3)xy,sqrt(3)yz,(2zz-xx-yy)/2,
   sqrt(3)xz,sqrt(3)(xx-yy)/2 before primitive normalization. */
typedef LIBGTO_ENUM{LIBGTO_PRECISION_AUTO = 0,
                    LIBGTO_PRECISION_REFERENCE = 1} libgto_precision_policy;
typedef struct {
  uint64_t wide_calls, fp64_calls, fp64_fallbacks, fp64_primitive_quartets;
} libgto_precision_statistics;
typedef struct libgto_pair libgto_pair;
typedef struct libgto_workspace libgto_workspace;
typedef struct libgto_batch libgto_batch;
typedef struct {
  libgto_representation representation;
  libgto_contraction_order contraction_order;
  int profile; /* nonzero collects phase times and may select a less fused path */
} libgto_eri_options;
typedef struct {
  size_t dimensions[4], elements, output_bytes, workspace_bytes;
} libgto_requirements;
typedef struct {
  double recurrence_seconds, contraction_seconds, transform_seconds, output_seconds;
  uint64_t primitive_quartets;
} libgto_statistics;
/* Pair creation deep-copies inputs. Immutable, thread-safe handles; no screening.
   Input/geometry/coefficients changes require a new handle. Do not reuse future
   coefficient-dependent bounds or task lists after changing coefficients. */
libgto_status libgto_pair_create(const libgto_shell *, const libgto_shell *, libgto_pair **);
void libgto_pair_destroy(libgto_pair *);
size_t libgto_pair_bytes(const libgto_pair *);
libgto_eri_options libgto_default_eri_options(void);
libgto_status libgto_eri_requirements(const libgto_pair *, const libgto_pair *,
                                      const libgto_eri_options *, libgto_requirements *);
libgto_status libgto_workspace_create(libgto_workspace **);
void libgto_workspace_destroy(libgto_workspace *);
libgto_status libgto_workspace_reserve(libgto_workspace *, const libgto_pair *, const libgto_pair *,
                                       const libgto_eri_options *);
/* Owned numeric storage, including allocated fixed-precision reconstruction
   scratch. Excludes handle/container/allocator overhead, immutable pairs and
   ordinary scalar call frames. Requirements include reconstruction storage when
   the immutable pair features select that AUTO path. */
size_t libgto_workspace_bytes(const libgto_workspace *);
/* Cumulative precision-path counters, including extra work before fallback.
   Pair preparation retains wide arithmetic; auxiliary precision is selected
   independently from angular and channel contraction precision.
   The local roundoff indicator is conservative in tested cases, not a rigorous
   certificate for every possible input. Wide projected fallback preserves
   semantics; it is distinct from the explicit original REFERENCE policy. */
/* Set policies only while the workspace/executor is idle. The reference policy
   uses unmerged raw wide evaluation. Low kernels transform pure components
   late; high-angular kernels may project E before contraction. */
libgto_status libgto_workspace_set_precision(libgto_workspace *, libgto_precision_policy);
libgto_status libgto_batch_set_precision(libgto_batch *, libgto_precision_policy);
libgto_status libgto_workspace_precision_statistics(const libgto_workspace *,
                                                    libgto_precision_statistics *);
libgto_status libgto_batch_precision_statistics(const libgto_batch *,
                                                libgto_precision_statistics *);
/* Cumulative local fixed-precision reconstruction attempts, including failures.
   Query only while the workspace/batch is idle. These are not error certificates. */
libgto_status libgto_workspace_reconstruction_count(const libgto_workspace *, uint64_t *);
libgto_status libgto_batch_reconstruction_count(const libgto_batch *, uint64_t *);
libgto_status libgto_workspace_statistics(const libgto_workspace *, libgto_statistics *);
/* Complete chemists' (ab|cd) tensor; i+nA*(j+nB*(k+nC*l)); axis index is
   component+ncomponent*channel. output_capacity is in doubles. Caller owns
   output; it survives workspace reuse. All functions return errors, no C++
   exceptions cross this ABI. On failure output content is unspecified. */
libgto_status libgto_eri(const libgto_pair *, const libgto_pair *, const libgto_eri_options *,
                         libgto_workspace *, double *, size_t output_capacity);
/* Borrowed complete block: expires at the next use/reserve/destroy of this
   workspace. Independent workspaces are required for concurrent calls. */
libgto_status libgto_eri_view(const libgto_pair *, const libgto_pair *, const libgto_eri_options *,
                              libgto_workspace *, const double **, size_t *);
/* The Boys API deliberately supports orders 0..8, T finite and nonnegative. */
libgto_status libgto_boys(unsigned max_order, double T, double *, size_t capacity);
const char *libgto_status_string(libgto_status);
typedef struct {
  const libgto_pair *ab, *cd;
  libgto_eri_options options;
  double *output;
  size_t output_capacity;
  uint64_t id;
} libgto_eri_task;
typedef struct {
  int threads;
  size_t max_tile_tasks, max_tile_output_bytes, max_workspace_bytes;
} libgto_batch_options;
libgto_batch_options libgto_default_batch_options(void);
libgto_status libgto_batch_create(const libgto_batch_options *, libgto_batch **);
void libgto_batch_destroy(libgto_batch *);
/* Bounded task parallelism within each submitted batch, no global thread pool. Tasks/output
   are caller-owned and must remain alive during the call. Output regions must
   be disjoint; inputs may share immutable pairs. The executor is non-reentrant.
   Oversized output blocks run alone; the per-worker numeric workspace cap is
   hard. Empty batches are valid. Task IDs are metadata, not a screening rule. */
libgto_status libgto_batch_compute(libgto_batch *, const libgto_eri_task *, size_t count);
size_t libgto_batch_workspace_bytes(const libgto_batch *);
/* Idle-time bounded binding configuration. count==0 uses the legacy two-pass
   path. Larger submissions also retain that compatibility path. Numeric scratch
   remains subject to max_workspace_bytes; metadata has this separate hard cap.
   bytes must cover the fixed plan cache as reported by binding_bytes initially.
   On bounded calls overlapping written output spans are rejected before writes. */
libgto_status libgto_batch_set_binding_limits(libgto_batch *, size_t count, size_t bytes);
size_t libgto_batch_binding_bytes(const libgto_batch *);
typedef struct {
  uint64_t preflight_tasks, plan_builds, plan_hits, bindings, scalar_reentries, legacy_batches;
} libgto_binding_statistics;
libgto_status libgto_batch_binding_statistics(const libgto_batch *, libgto_binding_statistics *);

#undef LIBGTO_ENUM
#ifdef __cplusplus
}
#endif
#endif
