# Copyright 2026 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Compare actual source revisions with the same saved model and sample arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import flax.serialization
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

from benchmark_unique import build_case, digest


def prepare(args):
    vs, H = build_case(args.case, args.samples, 17)
    x = vs.samples.reshape(-1, vs.hilbert.size)
    xp, mels = H.get_conn_padded(x)
    directory = args.inputs / args.case
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "variables.msgpack").write_bytes(
        flax.serialization.to_bytes(vs.variables)
    )
    np.save(directory / "samples.npy", np.asarray(x))
    rows = np.concatenate((np.asarray(x), np.asarray(xp).reshape(-1, vs.hilbert.size)))
    metadata = dict(
        case=args.case,
        seed=17,
        samples=len(x),
        sites=vs.hilbert.size,
        variables_sha256=digest(vs.variables),
        samples_sha256=digest(x),
        connectivity_sha256=digest((xp, mels)),
        parameter_dtypes=sorted({str(v.dtype) for v in jax.tree.leaves(vs.parameters)}),
        counts=dict(
            padded=len(rows),
            compact=len(x) + int(jnp.any(xp != x[:, None], axis=-1).sum()),
            unique=len(np.unique(rows, axis=0)),
            unique_references=len(np.unique(np.asarray(x), axis=0)),
        ),
    )
    assert set(metadata["parameter_dtypes"]) <= {"float64", "complex128"}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("prepared", args.case, metadata["samples_sha256"], flush=True)


def measure(args):
    directory = args.inputs / args.case
    metadata = json.loads((directory / "metadata.json").read_text())
    vs, H = build_case(args.case, metadata["samples"], 17)
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
    if args.variant == "main":
        expected_kernel = "local_value_kernel_jax_chunked"
    elif args.variant == "pr2293":
        nk.config.netket_experimental_flattened_kernel = True
        expected_kernel = "local_value_kernel_jax_flattened"
    else:
        nk.config.netket_experimental_flattened_kernel = False
        nk.config.netket_experimental_unique_kernel = True
        expected_kernel = "local_value_kernel_jax_fingerprint"
    kernel = nk.vqs.get_local_kernel(vs, H, args.chunk)
    assert kernel.__name__ == expected_kernel, kernel
    source_file = Path(nk.__file__).parent / "vqs/mc/kernels.py"
    report = dict(
        **metadata,
        variant=args.variant,
        revision=args.revision,
        netket_source=str(Path(nk.__file__).resolve()),
        kernel=kernel.__name__,
        kernel_source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        jax_version=jax.__version__,
        device=str(jax.devices()[0]),
        chunk_size=args.chunk,
        seconds=[],
        output_hashes=[],
    )
    start = time.perf_counter()
    run = (
        jax.jit(lambda v, x, H: kernel(vs._apply_fun, v, x, H, chunk_size=args.chunk))
        .lower(vs.variables, x, H)
        .compile()
    )
    report["compile_s"] = time.perf_counter() - start
    for _ in range(10):
        value = jax.block_until_ready(run(vs.variables, x, H))
    for _ in range(args.repeats):
        start = time.perf_counter()
        value = jax.block_until_ready(run(vs.variables, x, H))
        report["seconds"].append(time.perf_counter() - start)
        report["output_hashes"].append(digest(value))
    report.update(
        median_ms=1e3 * float(np.median(report["seconds"])),
        p10_ms=1e3 * float(np.percentile(report["seconds"], 10)),
        p90_ms=1e3 * float(np.percentile(report["seconds"], 90)),
        distinct_outputs=len(set(report["output_hashes"])),
    )
    # Check that the selected kernel is the one exercised by the public API.
    vs._samples = x.reshape(vs.sampler.n_chains, -1, vs.hilbert.size)
    public = np.asarray(vs.local_estimators(H, chunk_size=args.chunk).data).reshape(-1)
    np.testing.assert_allclose(value, public, rtol=1e-12, atol=1e-10)
    report["public_api_agrees"] = True
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"{args.case}-{args.variant}"
    np.save(stem.with_suffix(".npy"), np.asarray(value))
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["distinct_outputs"] == 1
    print(
        args.case,
        args.variant,
        f"{report['median_ms']:.6f} ms",
        f"compile {report['compile_s']:.3f} s",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--variant", choices=("main", "pr2293", "candidate", "published")
    )
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 and jax.device_count() == 1
    (prepare if args.prepare else measure)(args)
