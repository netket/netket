# Copyright 2026 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Reproducible FP64 local-energy benchmarks using public NetKet models.

Examples:
  JAX_ENABLE_X64=1 python benchmark_unique.py --case heisenberg16 --output result.json
  JAX_ENABLE_X64=1 python benchmark_unique.py --case vit4 --samples 512 --check-gradients

Heisenberg/RBMSymm follows Examples/Heisenberg1d/heisenberg1d.py; Hubbard/Slater
follows Examples/Fermions/fermi_hubbard.py; ViT follows the public tutorial.
Small finite Hilbert spaces expose repeated configurations; larger cases and
cheap Ising/RBM provide controls for the cost of sorting. All sizes and actual
duplicate counts are reported. No samples are manually duplicated.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from netket.vqs.mc import kernels


def build_case(name, samples, seed):
    if name.startswith("heisenberg"):
        n = int(name.removeprefix("heisenberg"))
        graph = nk.graph.Chain(n)
        hi = nk.hilbert.Spin(0.5, n, total_sz=0)
        H = nk.operator.Heisenberg(hi, graph)
        model = nk.models.RBMSymm(
            symmetries=graph.translation_group(),
            alpha=4,
            use_visible_bias=False,
            param_dtype=jnp.float64,
        )
        sampler = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=32)
    elif name == "ising20":
        graph = nk.graph.Chain(20)
        hi = nk.hilbert.Spin(0.5, 20)
        H = nk.operator.IsingJax(hi, graph, h=1.0)
        model = nk.models.RBM(alpha=2, param_dtype=jnp.float64)
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=32)
    elif name.startswith("hubbard"):
        side = int(name.removeprefix("hubbard"))
        graph = nk.graph.Square(side)
        n = graph.n_nodes
        hi = nk.hilbert.SpinOrbitalFermions(
            n, s=0.5, n_fermions_per_spin=(n // 2, n // 2)
        )
        H = nk.operator.FermiHubbardJax(hi, graph, t=1.0, U=4.0)
        model = nk.models.Slater2nd(hi, param_dtype=jnp.float64)
        sampler = nk.sampler.MetropolisFermionHop(hi, graph=graph, n_chains=32)
    else:
        from vit import ViT

        side = int(name.removeprefix("vit"))
        graph = nk.graph.Square(side, max_neighbor_order=2)
        hi = nk.hilbert.Spin(0.5, graph.n_nodes, total_sz=0)
        H = nk.operator.Heisenberg(hi, graph, J=[1.0, 0.5], sign_rule=[False, False])
        model = ViT(
            num_layers=4, d_model=60, n_heads=10, patch_size=2, transl_invariant=True
        )
        sampler = nk.sampler.MetropolisExchange(hi, graph=graph, d_max=2, n_chains=32)
    vs = nk.vqs.MCState(
        sampler, model, n_samples=samples, n_discard_per_chain=16, seed=seed
    )
    return vs, H.to_jax_operator()


def digest(tree):
    h = hashlib.sha256()
    for value in jax.tree.leaves(tree):
        a = np.asarray(value)
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def benchmark(args):
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "Set JAX_ENABLE_X64=1: this benchmark requires double precision."
        )
    if jax.device_count() != 1:
        raise RuntimeError(
            "Run this timing example on one device; sharding is covered by the tests."
        )
    vs, H = build_case(args.case, args.samples, args.seed)
    x = vs.samples.reshape(-1, vs.hilbert.size)
    xp, mels = H.get_conn_padded(x)
    x_host, xp_host = np.asarray(x), np.asarray(xp)
    all_rows = np.concatenate((x_host, xp_host.reshape(-1, vs.hilbert.size)))
    unique = len(np.unique(all_rows, axis=0))
    offdiag = int(np.any(xp_host != x_host[:, None], axis=-1).sum())
    dtypes = sorted({str(p.dtype) for p in jax.tree.leaves(vs.parameters)})
    assert set(dtypes) <= {"float64", "complex128"}, dtypes
    logs = vs._apply_fun(vs.variables, x)
    assert str(logs.dtype) in ("float64", "complex128"), logs.dtype
    report = dict(
        case=args.case,
        seed=args.seed,
        samples=int(len(x)),
        sites=vs.hilbert.size,
        devices=[str(d) for d in jax.devices()],
        jax_version=jax.__version__,
        parameter_dtypes=dtypes,
        log_dtype=str(logs.dtype),
        samples_sha256=digest(x),
        variables_sha256=digest(vs.variables),
        scope="Fixed seeded model and naturally sampled configurations; local estimators only, no optimization step",
        counts=dict(
            padded_rows=int(len(all_rows)),
            compact_rows=int(len(x) + offdiag),
            unique_rows=unique,
            unique_reference_rows=len(np.unique(x_host, axis=0)),
            duplicate_connection_slots_including_padding=int(
                sum(len(row) - len(np.unique(row, axis=0)) for row in xp_host)
            ),
        ),
        methods={},
    )
    functions, outputs = {}, {}
    methods = dict(
        padded=kernels.local_value_kernel_jax_chunked,
        compact=kernels.local_value_kernel_jax_flattened,
        unique=kernels.local_value_kernel_jax_unique,
    )
    for name, kernel in methods.items():
        start = time.perf_counter()
        run = (
            jax.jit(
                lambda v, x, H: kernel(vs._apply_fun, v, x, H, chunk_size=args.chunk)
            )
            .lower(vs.variables, x, H)
            .compile()
        )
        compile_s = time.perf_counter() - start
        for _ in range(2):
            value = jax.block_until_ready(run(vs.variables, x, H))
        outputs[name] = np.asarray(value)
        functions[name] = run
        report["methods"][name] = dict(
            compile_s=compile_s, seconds=[], output_hashes=[]
        )
        print(args.case, name, "compiled", round(compile_s, 3), flush=True)
    names = list(methods)
    for repeat in range(args.repeats):
        for name in names[repeat % 3 :] + names[: repeat % 3]:
            start = time.perf_counter()
            value = jax.block_until_ready(functions[name](vs.variables, x, H))
            report["methods"][name]["seconds"].append(time.perf_counter() - start)
            report["methods"][name]["output_hashes"].append(digest(value))
    for name, actual in outputs.items():
        row = report["methods"][name]
        row.update(
            median_s=float(np.median(row["seconds"])),
            max_abs_error=float(np.max(np.abs(actual - outputs["padded"]))),
            passes_fp64_check=bool(
                np.allclose(actual, outputs["padded"], rtol=1e-12, atol=1e-10)
            ),
            distinct_outputs=len(set(row["output_hashes"])),
        )
        print(
            args.case,
            name,
            "median",
            row["median_s"],
            "error",
            row["max_abs_error"],
            flush=True,
        )
    # Preserve timings even if a later diagnostic fails.
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.check_gradients:
        gradients = {}
        small_x = x[:16]
        for name in ("padded", "unique"):
            kernel = methods[name]

            def loss(p, samples, op):
                variables = {**vs.variables, "params": p}
                value = kernel(
                    vs._apply_fun, variables, samples, op, chunk_size=args.chunk
                )
                return jnp.mean(jnp.abs(value) ** 2)

            grad = jax.jit(jax.grad(loss))
            values = [
                jax.block_until_ready(grad(vs.parameters, small_x, H)) for _ in range(5)
            ]
            gradients[name] = values[0]
            report["methods"][name]["distinct_gradients"] = len(
                {digest(v) for v in values}
            )
            report["methods"][name]["gradient_repeat_max_abs_delta"] = max(
                float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                for v in values
                for a, b in zip(jax.tree.leaves(v), jax.tree.leaves(values[0]))
            )
            for v in values:
                jax.tree.map(
                    lambda a, b: np.testing.assert_allclose(
                        a, b, rtol=1e-11, atol=1e-9
                    ),
                    v,
                    values[0],
                )
        differences = jax.tree.leaves(
            jax.tree.map(
                lambda a, b: jnp.max(jnp.abs(a - b)),
                gradients["padded"],
                gradients["unique"],
            )
        )
        report["gradient_max_abs_error"] = float(max(map(float, differences)))
        try:
            jax.tree.map(
                lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-9),
                gradients["padded"],
                gradients["unique"],
            )
            report["gradients_agree"] = True
        except AssertionError:
            report["gradients_agree"] = False
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))
    assert all(
        r["passes_fp64_check"] and r["distinct_outputs"] == 1
        for r in report["methods"].values()
    )
    if args.check_gradients:
        assert report["gradients_agree"]
        # Record bitwise repeatability separately: model/backend gradients may
        # themselves be nondeterministic, including in the padded reference.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "heisenberg16",
            "heisenberg20",
            "ising20",
            "hubbard2",
            "hubbard4",
            "vit4",
            "vit8",
        ),
        default="heisenberg16",
    )
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--check-gradients", action="store_true")
    parser.add_argument("--output", type=Path)
    benchmark(parser.parse_args())
