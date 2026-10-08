# Compaction and duplicate reuse in local estimators

This is a standalone implementation against main, incorporating the initial
compaction work from PR #2293 with attribution. It does not require #2293,
#2294 or #2295 to be merged, new operator methods, or per-operator bounds.

This benchmark uses seeded models and Monte Carlo samples from standard public
NetKet examples. It times local estimators with all connectivity preparation
included. It does not measure a complete optimization step.

```bash
JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
  python Examples/LocalEnergy/benchmark_unique.py \
  --case heisenberg16 --samples 2048 --check-gradients --output heisenberg.json
```

Cases:

| Case | Public model and Hamiltonian | Purpose |
|---|---|---|
| `heisenberg16`, `heisenberg20`, `heisenberg64` | Heisenberg chain and `RBMSymm`, following `Examples/Heisenberg1d/heisenberg1d.py` | Padding and naturally repeated states, including a larger Hilbert space control |
| `ising20` | Transverse-field Ising and `RBM` | Cheap model with little padding or duplication; measures overhead |
| `hubbard2`, `hubbard4` | Fermi–Hubbard and `Slater2nd`, following `Examples/Fermions/fermi_hubbard.py` | Fermionic signs, finite Hilbert space reuse, and a larger system control |
| `vit4`, `vit8` | J1–J2 and the public ViT tutorial model, 4 layers, width 60, 10 heads, patch size 2 | Costlier model, with explicitly reported lattice sizes |

`vit.py` copies the model definitions from
`docs/tutorials/ViT-wave-function.ipynb`; use `--samples 512` initially for these
cases. The small Hilbert spaces are deliberate demonstrations of reuse, not
evidence that larger systems have the same duplicate fraction. Samples are
produced by the sampler; none are manually repeated to inflate the benefit.

The report compares padded evaluation, compaction, and exact deduplication.
The compact paths use coarse tails by default: for chunk size 128, tail
batches stop at 16 rows and evaluate at most 15 extra valid configurations.
Their outputs are discarded. Direct kernel calls can select exact tails with
`min_chunk_size=1`. It records reference and connected row counts,
unique rows per device, parameter/output dtypes, input hashes, compile time,
warmed synchronous timings, numerical differences, and repeated output hashes.
`--check-gradients` additionally checks values and gradient repeatability on
the first 16 samples. It does not time gradient evaluation.
Bitwise gradient hashes are reported separately from numerical agreement:
the model's own backward pass may vary even in the padded baseline. The
deduplication algorithm's expansion is tested independently with analytic
inputs to isolate it from model-level variability.

All parameters and model arithmetic use double precision. Local values must
agree at `rtol=1e-12, atol=1e-10`; a failing check exits unsuccessfully after
saving the report. Gradient checks use `rtol=1e-11, atol=1e-9`. These tolerances
test agreement between the implementations; they are not a certificate of
absolute accuracy for an ill-conditioned wavefunction.

The benchmark runs on one CPU or GPU device. Sharded correctness and the
absence of new collectives are covered separately in the test suite.
Sorting overhead can exceed the saved network work, so all cases—including
regressions—should be included when reporting results.

## Cheap reuse heuristic

`benchmark_reuse.py` additionally compares full lexicographic grouping, sorting
32-bit fingerprints with exact row checks, and the experimental sample-reuse
gate. The gate tries fingerprint grouping only if at least one eighth of the
reference samples repeat on that device; otherwise it uses compaction. It can
miss reuse but cannot merge unequal states. The sample gate is a benchmark
alternative: its extra cost often outweighs its benefit once fingerprint
grouping is cheap. The opt-in MCState flag selects unconditional fingerprint
grouping. The flag remains disabled by default. No timing tuner is involved.

```bash
JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
  python Examples/LocalEnergy/benchmark_reuse.py \
  --case heisenberg64 --samples 2048 --output reuse.json
```

The report includes planning cost, the gate decision, actual row counts,
compile time, ten synchronous interleaved measurements, FP64 agreement and
gradient checks. The 64-site chain supplements the smaller reuse examples;
repeat frequency depends on the sampled distribution, not only lattice size.
These remain seeded initial-model benchmarks, not complete training steps.

## Comparing source revisions

The compact method in the algorithm benchmarks above is the implementation in
this branch. It is not unmodified PR #2293. The historical source comparisons
below measured the earlier implementation published in #2296 at `b9dc149`.
They use
upstream main at `5e4511b7883d89d83b0aba534b37c3c672a15a42`, original PR #2293 at
`edeaf94d2768384e36f9db9480de54b5efbe8866`, and the earlier version of this PR at
`d45f0e76c6898879cc7f278af701c03845c2a64e` as separate reference checkouts.
Their model, operator and dependency code is identical.

`benchmark_revisions.py` prepares one saved model and sample batch. Run the
same script under each imported source (`PYTHONPATH`) to measure the actual
MCState-selected kernel and save an independent reference. Use `--variant main`,
`pr2293` or `published` for the three baselines; the published variant enables
fingerprint reuse. Each run checks parameter, sample and connectivity hashes
and agreement with `MCState.local_estimators`.

```bash
# Run the scripts from this branch, using paths to the requested source checkouts.
export JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest
unset NETKET_EXPERIMENTAL_FLATTENED_KERNEL NETKET_EXPERIMENTAL_UNIQUE_KERNEL
PYTHONPATH=/path/to/main python Examples/LocalEnergy/benchmark_revisions.py \
  --prepare --case heisenberg64 --samples 2048 --inputs /tmp/eloc-inputs
PYTHONPATH=/path/to/main python Examples/LocalEnergy/benchmark_revisions.py \
  --case heisenberg64 --inputs /tmp/eloc-inputs --variant main \
  --revision 5e4511b --output /tmp/eloc-references
PYTHONPATH=/path/to/pr2293 python Examples/LocalEnergy/benchmark_revisions.py \
  --case heisenberg64 --inputs /tmp/eloc-inputs --variant pr2293 \
  --revision edeaf94d --output /tmp/eloc-references
PYTHONPATH=/path/to/previous-pr python Examples/LocalEnergy/benchmark_revisions.py \
  --case heisenberg64 --inputs /tmp/eloc-inputs --variant published \
  --revision d45f0e7 --output /tmp/eloc-references
```

Then `benchmark_interleaved.py` times all five kernels in rotating order in a
single process. It imports the three unmodified reference kernel modules from
`--sources/main`, `--sources/pr2293`, and `--sources/candidate` (the previous PR
snapshot). Their hashes and outputs must match the independently verified
reference files. The two candidate implementations are also checked through that revision's
MCState dispatcher and public API. All checkouts need generated
version metadata and common installed dependencies; an editable installation
generates the metadata.

```bash
PYTHONPATH=/path/to/this-branch python Examples/LocalEnergy/benchmark_interleaved.py \
  --case heisenberg64 --inputs /tmp/eloc-inputs --sources /path/to/baselines \
  --reference-results /tmp/eloc-references --output /tmp/eloc-results
```

The following results use one H100, JAX 0.10.1, chunk size 128, ten warmups and
21 synchronized interleaved measurements per method. Parameters and model
arithmetic use FP64, with 2,048 samples for spin/Hubbard and 512 for ViT. Values
are median milliseconds per local-energy evaluation; compilation, sampling
and optimization are excluded. Main uses its default kernel, #2293 enables
its original flattened kernel, and the previous PR column enables its
original fingerprint kernel.

| Example | Current main (ms) | Original #2293 (ms) | Previous dedup (ms) | Fixed compact (ms) | Fixed dedup (ms) |
|---|---:|---:|---:|---:|---:|
| ising20 | 5.067 | 6.597 | 6.919 | 3.419 | 3.384 |
| heisenberg20 | 12.558 | 2.928 | 0.862 | 1.939 | 0.828 |
| heisenberg64 | 87.381 | 9.281 | 2.288 | 4.325 | 1.561 |
| hubbard2 | 13.922 | 3.149 | 0.626 | 1.864 | 0.631 |
| hubbard4 | 423.654 | 29.005 | 29.462 | 23.154 | 23.274 |
| vit4 | 90.999 | 18.938 | 4.458 | 17.830 | 4.754 |
| vit8 | 159.730 | 88.880 | 89.774 | 82.711 | 82.915 |

Both candidate kernels at that revision beat both pinned upstream baselines
in every measured case.
The fixed-block loop removes the Ising20, Hubbard4 and ViT8 regressions.
Relative to the previous fingerprint implementation, the tiny Hubbard2 case
is within 1%, while ViT4 is about 6.6% slower in this run; ViT4 remains about
4× faster than original #2293. These differences are retained in the table.
Reuse depends on the sampled distribution: Heisenberg64 has only 376 distinct
reference samples among 2,048. The 2×2 Hubbard graph follows NetKet's
length-two boundary behavior in every source. These seeded initial models do
not establish speedups for all optimized states or whole VMC steps.

## Historical GPU loop overhead and numerical checks

The old compact evaluator used a loop whose trip count depended on the input
batch. A dense fixed-length scan could be faster despite doing more network
work. Packing rows alone made only a small difference. The implementation
measured in the table above ran full chunks in binary blocks: 320 chunks,
for example, used blocks of 256 and 64 iterations. The native-AD update below
bounds the sum of all block capacities as well as the work actually executed. Runtime conditions select occupied blocks; no padded network rows
are evaluated. Selected rows are packed once and sliced contiguously. The
block-selection phase is bypassed when only a partial chunk is occupied.
This is shared by plain compaction and deduplication, without a timing tuner
or operator-specific branch.

`benchmark_results.json` records individual timings, source/input hashes,
compilation costs, executable temporary-memory estimates, duplicate counts
and all pairwise errors. Compilation and temporary storage can increase;
these timings describe warmed execution. All 70 pairwise forward comparisons
pass `rtol=1e-12, atol=1e-10`, with a maximum difference of 2.58e-12. All 21
forward outputs repeat bitwise within each method.

`check_revision_gradients.py` checks the first 16 saved samples of each model
against all three original sources. All seven cases pass
`rtol=1e-11, atol=1e-9`; the largest gradient difference versus main is
1.51e-10. Model-gradient bitwise determinism is not claimed. The kernel tests
also check exact network-row counts across block boundaries, no retracing,
zero-coefficient derivatives, mixed second derivatives and sharding.

## Historical native differentiation and compilation (7 October, `80c8d916`)

The evaluator at `80c8d916` incorporated the network JIT wrapper and exact-capacity
block schedule from PR #2293 at `c87a3c77`. It uses native differentiation in
place of the earlier custom-JVP scan. Exact tails, row packing and optional
fingerprint reuse are retained. These changes require no new operator API.
The sum of the full-block capacities equals the allocated full-chunk capacity,
limiting reverse-mode residual storage for skipped blocks.

The baselines are main `5e4511b`, our published PR `b9dc149`, and latest #2293
`c87a3c77`. Both the default coarse tail and explicit `min_chunk_size=1` of
#2293 are measured. The implementation at `80c8d916` retained exact tails;
the current branch uses coarse tails by default, as documented below.

One H100, JAX/jaxlib 0.10.1, FP64 parameters and model arithmetic. Each
(case, source, mode, chunk size) runs in a fresh process, with persistent
compilation caching disabled and in-memory JAX caches cleared after common
model setup. The report separates tracing/lowering from XLA compilation.
Execution timings use ten warmups and 21 synchronized calls; source order is
rotated by case. Source, parameter, sample and connectivity hashes are checked.
All sources use identical public models and saved inputs. Compilation excludes
model setup, sampling and input validation. These are local-energy kernels,
not entire VMC steps.

Forward runs use 2,048 samples for spin/Hubbard and 512 for ViT. Gradients of
mean(abs(E_loc)**2) use all 2,048 samples for spin/Hubbard and the first 16 saved
samples for ViT. The derivative speedups apply when differentiating local
energies; the usual covariance-based VMC gradient follows a different path.
Memory numbers are XLA executable temporary-memory estimates, not measured
peak resident GPU usage.

Forward runtime, milliseconds (chunk 128):

| Case | Main | Published compact | Published dedup | Latest #2293 coarse | Latest #2293 exact | Updated compact | Updated dedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| ising20 | 5.176 | 3.374 | 3.327 | 3.307 | 3.433 | 3.402 | 3.377 |
| heisenberg20 | 12.742 | 1.936 | 0.805 | 1.893 | 1.968 | 1.945 | 0.808 |
| heisenberg64 | 86.759 | 4.132 | 1.375 | 4.145 | 4.256 | 4.107 | 1.383 |
| hubbard2 | 13.779 | 1.851 | 0.612 | 1.808 | 1.871 | 1.841 | 0.613 |
| hubbard4 | 437.428 | 23.277 | 23.044 | 23.590 | 23.780 | 22.931 | 23.271 |
| vit4 | 91.141 | 17.317 | 4.110 | 17.281 | 17.417 | 17.307 | 4.076 |
| vit8 | 159.473 | 83.507 | 83.578 | 82.858 | 83.591 | 84.504 | 83.792 |

Cold forward trace + compile, seconds (chunk 128):

| Case | Main | Published compact | Published dedup | Latest #2293 coarse | Latest #2293 exact | Updated compact | Updated dedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| ising20 | 0.749 | 1.591 | 2.103 | 1.377 | 1.702 | 1.535 | 2.097 |
| heisenberg20 | 0.903 | 1.968 | 2.293 | 1.451 | 1.973 | 1.825 | 2.346 |
| heisenberg64 | 1.691 | 2.641 | 2.981 | 1.752 | 2.205 | 2.140 | 2.704 |
| hubbard2 | 1.429 | 2.381 | 3.434 | 1.944 | 2.454 | 2.380 | 3.317 |
| hubbard4 | 0.955 | 3.499 | 4.160 | 2.896 | 3.420 | 3.382 | 4.129 |
| vit4 | 2.855 | 14.249 | 14.511 | 8.796 | 12.903 | 13.282 | 13.676 |
| vit8 | 2.518 | 15.005 | 15.982 | 9.467 | 13.911 | 13.847 | 14.597 |

Gradient runtime, milliseconds (chunk 128; ViT uses 16 samples):

| Case | Main | Published compact | Published dedup | Latest #2293 coarse | Latest #2293 exact | Updated compact | Updated dedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| ising20 | 19.308 | 30.883 | 32.335 | 11.746 | 11.828 | 12.284 | 12.189 |
| heisenberg20 | 24.721 | 29.663 | 30.313 | 5.265 | 5.391 | 5.527 | 1.846 |
| heisenberg64 | 156.076 | 91.148 | 89.766 | 12.927 | 13.038 | 13.351 | 4.331 |
| hubbard2 | 21.246 | 16.773 | 18.037 | 5.145 | 5.320 | 5.295 | 1.095 |
| hubbard4 | 737.602 | 147.536 | 149.009 | 78.568 | 77.018 | 77.730 | 76.043 |
| vit4 | 18.581 | 18.579 | 18.916 | 8.810 | 10.139 | 10.645 | 7.898 |
| vit8 | 52.882 | 81.103 | 80.454 | 34.528 | 36.914 | 36.918 | 38.413 |

Cold gradient trace + compile, seconds:

| Case | Main | Published compact | Published dedup | Latest #2293 coarse | Latest #2293 exact | Updated compact | Updated dedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| ising20 | 1.536 | 2.878 | 3.542 | 3.228 | 3.905 | 4.000 | 4.889 |
| heisenberg20 | 1.934 | 3.302 | 3.960 | 3.550 | 4.303 | 4.301 | 5.101 |
| heisenberg64 | 1.535 | 3.437 | 4.864 | 3.990 | 4.940 | 4.834 | 6.465 |
| hubbard2 | 1.882 | 2.695 | 4.438 | 3.514 | 4.071 | 4.286 | 6.304 |
| hubbard4 | 1.476 | 3.590 | 5.939 | 5.442 | 6.192 | 6.590 | 8.673 |
| vit4 | 9.249 | 37.053 | 35.368 | 27.060 | 41.141 | 43.570 | 42.300 |
| vit8 | 13.919 | 40.201 | 39.328 | 35.338 | 49.486 | 53.682 | 53.092 |

Gradient executable temporary memory, MiB:

| Case | Main | Published compact | Published dedup | Latest #2293 coarse | Latest #2293 exact | Updated compact | Updated dedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| ising20 | 49.716 | 50.706 | 51.967 | 51.044 | 51.070 | 50.937 | 51.860 |
| heisenberg20 | 90.179 | 92.898 | 94.331 | 92.395 | 92.411 | 91.921 | 93.348 |
| heisenberg64 | 860.364 | 896.104 | 906.725 | 873.062 | 873.157 | 864.064 | 875.257 |
| hubbard2 | 2.510 | 3.123 | 3.955 | 3.017 | 3.032 | 3.175 | 3.977 |
| hubbard4 | 152.736 | 163.916 | 168.852 | 156.592 | 156.603 | 157.622 | 160.744 |
| vit4 | 280.209 | 364.554 | 392.071 | 341.593 | 344.213 | 344.306 | 342.831 |
| vit8 | 4215.818 | 4515.700 | 4508.849 | 4411.080 | 4409.224 | 4416.989 | 4437.268 |

Chunk-4096 ViT control, cold forward trace + compile in seconds:

| Case | Main | Published compact | Published dedup | Latest #2293 coarse | Latest #2293 exact | Updated compact | Updated dedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| vit4 | 4.529 | 20.153 | 20.569 | 7.696 | 19.275 | 19.270 | 20.041 |
| vit8 | 4.568 | 20.531 | 21.246 | 7.934 | 19.832 | 19.693 | 21.055 |

Chunk-4096 ViT control, forward runtime in milliseconds:

| Case | Main | Published compact | Published dedup | Latest #2293 coarse | Latest #2293 exact | Updated compact | Updated dedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| vit4 | 6.038 | 3.691 | 1.917 | 3.002 | 3.631 | 3.670 | 1.916 |
| vit8 | 87.062 | 42.175 | 42.426 | 41.261 | 42.150 | 42.208 | 42.359 |

The backward execution gains come with higher backward compilation costs in
some cases. Exact tails continue to compile more shapes than the coarse-tail
alternative. Forward performance, compile time and memory are separate metrics;
none is claimed to improve universally. Dedup remains optional because its
benefit depends on repetition, and the full padded connectivity tensor is
still constructed. Both experimental flags remain disabled by default.

Reproduce a single measurement using `benchmark_native_ad.py` from this branch,
with `PYTHONPATH` selecting the requested source checkout. The source path and
SHA256 of its `netket/vqs/mc/kernels.py` are required arguments. For example:

```bash
JAX_ENABLE_X64=1 JAX_DEFAULT_MATMUL_PRECISION=highest \
PYTHONPATH=/path/to/this-branch python Examples/LocalEnergy/benchmark_native_ad.py \
  --case heisenberg64 --inputs /tmp/eloc-inputs --source /path/to/this-branch \
  --kernel-sha256 VERIFIED_KERNEL_SHA256 --variant candidate_dedup \
  --revision YOUR_REVISION --mode gradient --output /tmp/eloc-native-results
```

Use the saved-input preparation instructions above. Each command measures one
source/mode in its own process. The explicit exact-tail #2293 control calls its
kernel with `min_chunk_size=1`; other forward variants are also checked against
their revision's `MCState.local_estimators` API.

All 336 pairwise comparisons pass the stated tolerances. The maximum absolute forward difference is 3.41e-12, and the maximum absolute gradient difference is 4.37e-10. All forward outputs are bitwise repeatable within each method. Most full-model gradients show small run-to-run rounding variation, also present on main and the published baseline; the hashes and maximum differences are recorded. Whole-model gradient bitwise determinism is not claimed. The updated tests pass on four CPUs (99) and one H100 (98, with one expected multi-device skip).

## Coarse tails by default (8 October update)

The compact and optional reuse paths now adopt the coarse-tail policy from
PR #2293, with attribution. The minimum tail batch defaults to one eighth of
the resolved chunk size, rounded down to a power of two and at least one.
Chunk size 128 therefore compiles connected batches of 128, 64, 32 and 16
rows, and evaluates at most 15 discarded filler rows per device and call.
Direct kernel calls retain `min_chunk_size=1` for exact tails. Full-chunk
capacity, native differentiation, packing and optional fingerprint reuse are
preserved. No new operator API or timing tuner is needed.

One H100, JAX/jaxlib 0.10.1, FP64 parameters and model arithmetic.
Baselines: main `5e4511b`, our preceding exact-tail implementation `80c8d916`,
and PR #2293 `c87a3c77` with its default coarse tails. Each source/mode/case
runs in a fresh process on the same allocated GPU, with persistent compilation
caching disabled and in-memory caches cleared after model setup. Source,
parameter, sample and connectivity hashes are verified. Warm runtime is the
median of 21 synchronized calls after ten warmups; source order rotates by
case. Default forward paths are also checked through the actual MCState API.

Spin/Hubbard forward and gradient runs use 2,048 saved samples; ViT forward
uses 512, and its gradient runs use the same first 16 saved samples in every
source. Gradients are of `mean(abs(E_loc)**2)`. These measure local-energy
kernels and their derivatives, not complete VMC steps. XLA temporary-memory
estimates are not whole-process peak GPU residency.

Forward runtime, milliseconds:

| Case | Main | Previous exact compact | Latest #2293 coarse | Coarse compact | Coarse dedup |
|---|---:|---:|---:|---:|---:|
| ising20 | 5.054 | 3.355 | 3.378 | 3.367 | 3.275 |
| heisenberg20 | 12.672 | 1.923 | 1.888 | 1.867 | 0.728 |
| heisenberg64 | 86.720 | 4.144 | 4.100 | 4.110 | 1.322 |
| hubbard2 | 14.091 | 1.834 | 1.788 | 1.782 | 0.546 |
| hubbard4 | 418.622 | 23.395 | 23.122 | 23.303 | 23.193 |
| vit4 | 90.677 | 17.251 | 17.122 | 17.224 | 3.933 |
| vit8 | 160.136 | 83.688 | 82.750 | 82.835 | 82.859 |

Cold forward trace + compile, seconds:

| Case | Main | Previous exact compact | Latest #2293 coarse | Coarse compact | Coarse dedup |
|---|---:|---:|---:|---:|---:|
| ising20 | 0.726 | 1.538 | 1.310 | 1.216 | 1.784 |
| heisenberg20 | 0.816 | 1.778 | 1.439 | 1.349 | 1.984 |
| heisenberg64 | 0.960 | 2.175 | 1.747 | 1.558 | 2.234 |
| hubbard2 | 1.484 | 2.175 | 1.734 | 1.692 | 2.894 |
| hubbard4 | 0.926 | 3.241 | 2.709 | 2.667 | 3.313 |
| vit4 | 2.587 | 12.898 | 8.449 | 8.690 | 9.255 |
| vit8 | 2.370 | 13.742 | 9.027 | 9.395 | 10.316 |

Gradient runtime, milliseconds (ViT: 16 samples):

| Case | Main | Previous exact compact | Latest #2293 coarse | Coarse compact | Coarse dedup |
|---|---:|---:|---:|---:|---:|
| ising20 | 19.487 | 12.245 | 11.695 | 12.222 | 12.030 |
| heisenberg20 | 24.628 | 5.550 | 5.236 | 5.384 | 1.697 |
| heisenberg64 | 154.262 | 13.347 | 12.733 | 13.339 | 4.149 |
| hubbard2 | 21.474 | 5.236 | 5.126 | 5.117 | 0.930 |
| hubbard4 | 734.849 | 77.468 | 75.481 | 78.847 | 75.343 |
| vit4 | 18.558 | 10.523 | 8.473 | 8.809 | 5.555 |
| vit8 | 52.711 | 36.566 | 34.388 | 34.526 | 36.015 |

Cold gradient trace + compile, seconds:

| Case | Main | Previous exact compact | Latest #2293 coarse | Coarse compact | Coarse dedup |
|---|---:|---:|---:|---:|---:|
| ising20 | 1.560 | 3.715 | 2.962 | 2.928 | 4.042 |
| heisenberg20 | 1.853 | 4.177 | 3.399 | 3.442 | 4.469 |
| heisenberg64 | 1.465 | 4.974 | 3.941 | 3.889 | 5.516 |
| hubbard2 | 1.811 | 4.093 | 3.252 | 3.331 | 5.161 |
| hubbard4 | 1.437 | 6.203 | 4.950 | 5.214 | 7.282 |
| vit4 | 9.001 | 43.112 | 26.576 | 28.466 | 27.979 |
| vit8 | 13.646 | 52.260 | 33.836 | 36.677 | 36.751 |

Gradient executable temporary memory, MiB:

| Case | Main | Previous exact compact | Latest #2293 coarse | Coarse compact | Coarse dedup |
|---|---:|---:|---:|---:|---:|
| ising20 | 49.716 | 50.937 | 51.044 | 50.626 | 51.953 |
| heisenberg20 | 90.179 | 91.921 | 92.395 | 91.399 | 93.496 |
| heisenberg64 | 860.364 | 864.064 | 873.062 | 863.971 | 875.371 |
| hubbard2 | 2.510 | 3.175 | 3.017 | 3.019 | 3.970 |
| hubbard4 | 152.736 | 157.622 | 156.592 | 155.582 | 160.734 |
| vit4 | 280.209 | 344.306 | 341.398 | 341.604 | 339.552 |
| vit8 | 4215.818 | 4416.990 | 4411.062 | 4413.540 | 4433.653 |

Chunk-4096 ViT cold forward trace + compile, seconds:

| Case | Main | Previous exact compact | Latest #2293 coarse | Coarse compact | Coarse dedup |
|---|---:|---:|---:|---:|---:|
| vit4 | 4.487 | 19.303 | 7.605 | 7.735 | 8.526 |
| vit8 | 4.567 | 19.890 | 7.792 | 7.979 | 8.977 |

Chunk-4096 ViT forward runtime, milliseconds:

| Case | Main | Previous exact compact | Latest #2293 coarse | Coarse compact | Coarse dedup |
|---|---:|---:|---:|---:|---:|
| vit4 | 5.992 | 3.669 | 2.953 | 3.036 | 1.219 |
| vit8 | 86.892 | 42.120 | 41.201 | 41.231 | 41.462 |

All 160 pairwise comparisons across 80 measurements pass: forward
`rtol=1e-12, atol=1e-10` (maximum difference 3.41e-12);
gradients `rtol=1e-11, atol=1e-9` (maximum difference 4.42e-10).
Every forward result repeats bitwise over 21 calls. Full-model gradients can
have small run-to-run rounding variation, also present on main; their hashes
and maximum differences are retained. Whole-model backward bitwise determinism
is not claimed.

The five kernel test files pass on four CPU devices (145 tests) and H100
(144 tests, one expected multi-device skip). New tests count actual forward,
JVP and VJP model rows, cover non-power-of-two chunks and capacity boundaries,
verify discarded filler derivatives, and check all public tail options with
zero operator coefficients. Changing occupied counts does not retrace.

Use `benchmark_native_ad.py` with the source-selection commands above.
`benchmark_coarse_tail_results.json` retains this update's measurements,
source/input hashes, timings, compilation, memory and numerical checks;
`benchmark_native_ad_results.json` preserves the historical exact-tail run.
Coarse tails reduce the number of compiled shapes, but batch-shape-dependent
rounding remains possible, particularly with mixed-precision networks.
Optional reuse can cost more than it saves when configurations seldom repeat.
Both experimental flags remain disabled by default. Full padded connectivity
storage remains allocated and can still dominate memory at larger sizes.
