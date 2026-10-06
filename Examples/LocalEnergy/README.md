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

The report compares padded evaluation, compaction with exact tails, and exact
deduplication with exact tails. It records reference and connected row counts,
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
this branch. It is not unmodified PR #2293. The source comparisons use current
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
reference files. The two current implementations are also checked through the
current MCState dispatcher and public API. All checkouts need generated
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

Both updated kernels beat both upstream baselines in every measured case.
The fixed-block loop removes the Ising20, Hubbard4 and ViT8 regressions.
Relative to the previous fingerprint implementation, the tiny Hubbard2 case
is within 1%, while ViT4 is about 6.6% slower in this run; ViT4 remains about
4× faster than original #2293. These differences are retained in the table.
Reuse depends on the sampled distribution: Heisenberg64 has only 376 distinct
reference samples among 2,048. The 2×2 Hubbard graph follows NetKet's
length-two boundary behavior in every source. These seeded initial models do
not establish speedups for all optimized states or whole VMC steps.

## GPU loop overhead and numerical checks

The old compact evaluator used a loop whose trip count depended on the input
batch. A dense fixed-length scan could be faster despite doing more network
work. Packing rows alone made only a small difference. Full chunks now run
in statically sized blocks: 320 chunks, for example, use blocks of 256 and 64
iterations. Runtime conditions select occupied blocks; no padded network rows
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
