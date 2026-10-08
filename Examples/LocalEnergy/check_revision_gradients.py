# Copyright 2026 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Check the updated kernel's gradients against original source baselines."""
import argparse
import importlib.util
import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from netket.vqs.mc import kernels

from benchmark_unique import build_case, digest


parser = argparse.ArgumentParser()
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--sources", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
baselines = {}
for name in ("main", "pr2293", "candidate"):
    spec = importlib.util.spec_from_file_location(
        name, args.sources / name / "netket/vqs/mc/kernels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baselines[name] = module
methods = dict(
    main=baselines["main"].local_value_kernel_jax_chunked,
    pr2293=baselines["pr2293"].local_value_kernel_jax_flattened,
    published=baselines["candidate"].local_value_kernel_jax_fingerprint,
    optimized_flat=kernels.local_value_kernel_jax_flattened,
    optimized_dedup=kernels.local_value_kernel_jax_fingerprint,
)
report = {}
for case in (
    "ising20",
    "hubbard2",
    "hubbard4",
    "heisenberg20",
    "heisenberg64",
    "vit4",
    "vit8",
):
    directory = args.inputs / case
    metadata = json.loads((directory / "metadata.json").read_text())
    vs, H = build_case(case, metadata["samples"], 17)
    vs.variables = jax.tree.map(
        jnp.asarray,
        flax.serialization.from_bytes(
            vs.variables, (directory / "variables.msgpack").read_bytes()
        ),
    )
    x = jnp.asarray(np.load(directory / "samples.npy"))
    assert digest(vs.variables) == metadata["variables_sha256"]
    assert digest(x) == metadata["samples_sha256"]
    x = x[:16]
    row = dict(
        variables_sha256=digest(vs.variables),
        samples_sha256=metadata["samples_sha256"],
        subset_samples=16,
        methods={},
    )
    gradients = {}
    for name, kernel in methods.items():

        def loss(p, x, H):
            values = kernel(
                vs._apply_fun, {**vs.variables, "params": p}, x, H, chunk_size=128
            )
            return jnp.mean(jnp.abs(values) ** 2)

        grad = jax.block_until_ready(jax.jit(jax.grad(loss))(vs.parameters, x, H))
        gradients[name] = grad
        differences = [
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(gradients["main"]))
        ]
        for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(gradients["main"])):
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-9)
        row["methods"][name] = dict(
            max_abs_error_vs_main=max(differences),
            gradient_sha256=digest(grad),
            agrees=True,
        )
    report[case] = row
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("gradients", case, "passed", flush=True)
    jax.clear_caches()
