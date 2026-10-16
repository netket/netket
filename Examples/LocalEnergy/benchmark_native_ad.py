# Copyright 2026 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Measure one source, kernel and AD mode per fresh process on saved inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import flax
import flax.serialization
import jax
import jax.numpy as jnp
import jaxlib
import netket as nk
import numpy as np

from benchmark_unique import build_case, digest


def flatten_output(value):
    return np.concatenate([np.asarray(x).reshape(-1) for x in jax.tree.leaves(value)])


def measure(args):
    assert jax.config.jax_enable_x64 and jax.device_count() == 1
    jax.config.update("jax_enable_compilation_cache", False)
    source = Path(nk.__file__).resolve().parent
    assert source.parent == args.source.resolve(), source
    kernel_sha = hashlib.sha256((source / "vqs/mc/kernels.py").read_bytes()).hexdigest()
    assert kernel_sha == args.kernel_sha256
    directory = args.inputs / args.case
    metadata = json.loads((directory / "metadata.json").read_text())
    vs, H = build_case(args.case, metadata["samples"], metadata["seed"])
    vs.variables = jax.tree.map(
        jnp.asarray,
        flax.serialization.from_bytes(
            vs.variables, (directory / "variables.msgpack").read_bytes()
        ),
    )
    x = jnp.asarray(np.load(directory / "samples.npy"))
    assert digest(vs.variables) == metadata["variables_sha256"]
    assert digest(x) == metadata["samples_sha256"]
    assert digest(H.get_conn_padded(x)) == metadata["connectivity_sha256"]
    assert {str(p.dtype) for p in jax.tree.leaves(vs.parameters)} <= {
        "float64",
        "complex128",
    }
    if args.limit_samples:
        x = x[: args.limit_samples]
    if args.variant != "main":
        nk.config.netket_experimental_flattened_kernel = True
        if "dedup" in args.variant:
            nk.config.netket_experimental_unique_kernel = True
    kernel = nk.vqs.get_local_kernel(vs, H, args.chunk)
    expected = (
        "local_value_kernel_jax_chunked"
        if args.variant == "main"
        else (
            "local_value_kernel_jax_fingerprint"
            if "dedup" in args.variant
            else "local_value_kernel_jax_flattened"
        )
    )
    assert kernel.__name__ == expected
    kwargs = dict(chunk_size=args.chunk)
    if args.variant == "pr2293_exact":
        kwargs["min_chunk_size"] = 1

    def values(v, x, H):
        return kernel(vs._apply_fun, v, x, H, **kwargs)

    def loss(p, x, H):
        return jnp.mean(jnp.abs(values({**vs.variables, "params": p}, x, H)) ** 2)

    if args.mode == "forward":
        function, inputs = values, (vs.variables, x, H)
    else:
        function, inputs = jax.grad(loss), (vs.parameters, x, H)
    jax.block_until_ready(inputs)
    jax.clear_caches()
    start = time.perf_counter()
    lowered = jax.jit(function).lower(*inputs)
    traced = time.perf_counter()
    run = lowered.compile()
    compiled = time.perf_counter()
    memory = run.memory_analysis()
    report = dict(
        **metadata,
        variant=args.variant,
        revision=args.revision,
        mode=args.mode,
        measured_samples=int(len(x)),
        measured_samples_sha256=digest(x),
        chunk_size=args.chunk,
        kernel=kernel.__name__,
        kernel_source_sha256=kernel_sha,
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        flax_version=flax.__version__,
        device=str(jax.devices()[0]),
        device_kind=jax.devices()[0].device_kind,
        persistent_compilation_cache=False,
        trace_lower_s=traced - start,
        xla_compile_s=compiled - traced,
        compile_s=compiled - start,
        memory={
            key: getattr(memory, key)
            for key in (
                "argument_size_in_bytes",
                "temp_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
            )
        },
        seconds=[],
        output_hashes=[],
        max_repeat_abs_error=0.0,
        public_api_agrees=None,
    )
    for _ in range(args.warmups):
        value = jax.block_until_ready(run(*inputs))
    reference = flatten_output(value)
    assert np.all(np.isfinite(reference))
    for _ in range(args.repeats):
        start = time.perf_counter()
        value = jax.block_until_ready(run(*inputs))
        report["seconds"].append(time.perf_counter() - start)
        report["output_hashes"].append(digest(value))
        flat = flatten_output(value)
        np.testing.assert_allclose(flat, reference, rtol=1e-11, atol=1e-9)
        report["max_repeat_abs_error"] = max(
            report["max_repeat_abs_error"], float(np.max(np.abs(flat - reference)))
        )
    if args.mode == "forward" and args.variant != "pr2293_exact":
        vs._samples = x.reshape(vs.sampler.n_chains, -1, vs.hilbert.size)
        public = np.asarray(vs.local_estimators(H, chunk_size=args.chunk).data).reshape(
            -1
        )
        np.testing.assert_allclose(public, reference, rtol=1e-12, atol=1e-10)
        report["public_api_agrees"] = True
    report.update(
        median_ms=1e3 * float(np.median(report["seconds"])),
        p10_ms=1e3 * float(np.percentile(report["seconds"], 10)),
        p90_ms=1e3 * float(np.percentile(report["seconds"], 90)),
        distinct_outputs=len(set(report["output_hashes"])),
        output_dtype=str(reference.dtype),
        output_sha256=digest(reference),
        output_leaf_shapes=[list(x.shape) for x in jax.tree.leaves(value)],
    )
    if args.mode == "forward":
        assert report["distinct_outputs"] == 1
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"{args.case}-c{args.chunk}-{args.mode}-{args.variant}"
    np.save(stem.with_suffix(".npy"), reference)
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        args.case,
        args.chunk,
        args.mode,
        args.variant,
        f"compile={report['compile_s']:.3f}s",
        f"run={report['median_ms']:.3f}ms",
        f"temp={report['memory']['temp_size_in_bytes']/2**20:.1f}MiB",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--kernel-sha256", required=True)
    p.add_argument(
        "--variant",
        choices=(
            "main",
            "published_flat",
            "published_dedup",
            "pr2293_coarse",
            "pr2293_exact",
            "candidate_flat",
            "candidate_dedup",
        ),
        required=True,
    )
    p.add_argument("--revision", required=True)
    p.add_argument("--mode", choices=("forward", "gradient"), required=True)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--repeats", type=int, default=21)
    p.add_argument("--output", type=Path, required=True)
    measure(p.parse_args())
