# Copyright 2026 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Compare full grouping, fingerprint grouping and a cheap sample-reuse gate."""
import argparse
from functools import partial
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from netket.vqs.mc import kernels
from benchmark_unique import build_case, digest


def benchmark(args):
    assert jax.config.jax_enable_x64 and jax.device_count() == 1
    vs, H = build_case(args.case, args.samples, 17)
    x = vs.samples.reshape(-1, vs.hilbert.size)
    xp, _ = H.get_conn_padded(x)
    rows = jnp.concatenate((x, xp.reshape(-1, vs.hilbert.size)))
    dtypes = sorted({str(p.dtype) for p in jax.tree.leaves(vs.parameters)})
    assert set(dtypes) <= {"float64", "complex128"}
    report = dict(
        case=args.case,
        samples=len(x),
        sites=vs.hilbert.size,
        jax_version=jax.__version__,
        parameter_dtypes=dtypes,
        samples_sha256=digest(x),
        variables_sha256=digest(vs.variables),
        counts=dict(
            padded=len(rows),
            compact=len(x) + int(jnp.any(xp != x[:, None], axis=-1).sum()),
            unique=len(np.unique(np.asarray(rows), axis=0)),
            fingerprint=int(kernels._fingerprint_row_plan(rows)[-1]),
            unique_references=len(np.unique(np.asarray(x), axis=0)),
        ),
        heuristic_uses_dedup=bool(kernels._sample_reuse_predicate(x)),
        planning_ms={},
        methods={},
    )
    # Isolate planning costs as well as measuring the complete local estimator.
    for name, f, inputs in (
        ("exact", kernels._unique_row_plan, rows),
        ("fingerprint", kernels._fingerprint_row_plan, rows),
        ("sample_gate", kernels._sample_reuse_predicate, x),
    ):
        run = jax.jit(f).lower(inputs).compile()
        for _ in range(3):
            jax.block_until_ready(run(inputs))
        seconds = []
        for _ in range(9):
            start = time.perf_counter()
            jax.block_until_ready(run(inputs))
            seconds.append(time.perf_counter() - start)
        report["planning_ms"][name] = 1e3 * float(np.median(seconds))
    methods = dict(
        padded=kernels.local_value_kernel_jax_chunked,
        compact=kernels.local_value_kernel_jax_flattened,
        exact=kernels.local_value_kernel_jax_unique,
        fingerprint=partial(kernels.local_value_kernel_jax_reuse, check_samples=False),
        heuristic=kernels.local_value_kernel_jax_reuse,
    )
    functions, outputs = {}, {}
    for name, kernel in methods.items():
        start = time.perf_counter()
        run = (
            jax.jit(lambda v, x, H: kernel(vs._apply_fun, v, x, H, chunk_size=128))
            .lower(vs.variables, x, H)
            .compile()
        )
        row = dict(compile_s=time.perf_counter() - start, seconds=[], hashes=[])
        for _ in range(3):
            output = jax.block_until_ready(run(vs.variables, x, H))
        outputs[name] = np.asarray(output)
        functions[name] = run
        report["methods"][name] = row
        print(args.case, name, "compiled", row["compile_s"], flush=True)
    names = list(methods)
    for i in range(10):
        for name in names[i % len(names) :] + names[: i % len(names)]:
            start = time.perf_counter()
            output = jax.block_until_ready(functions[name](vs.variables, x, H))
            report["methods"][name]["seconds"].append(time.perf_counter() - start)
            report["methods"][name]["hashes"].append(digest(output))
    for name, row in report["methods"].items():
        row.update(
            median_ms=1e3 * float(np.median(row["seconds"])),
            max_abs_error=float(np.max(np.abs(outputs[name] - outputs["padded"]))),
            fp64_agreement=bool(
                np.allclose(outputs[name], outputs["padded"], rtol=1e-12, atol=1e-10)
            ),
            distinct_outputs=len(set(row["hashes"])),
        )
        print(
            args.case,
            name,
            row["median_ms"],
            "ms",
            "error",
            row["max_abs_error"],
            flush=True,
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    gradients = {}
    for name in ("compact", "fingerprint", "heuristic"):
        kernel = methods[name]

        def loss(p, x, H):
            values = kernel(
                vs._apply_fun, {**vs.variables, "params": p}, x, H, chunk_size=128
            )
            return jnp.mean(jnp.abs(values) ** 2)

        grad = jax.jit(jax.grad(loss))
        values = [
            jax.block_until_ready(grad(vs.parameters, x[:16], H)) for _ in range(3)
        ]
        gradients[name] = values[0]
        report["methods"][name]["distinct_gradients"] = len({digest(v) for v in values})
        report["methods"][name]["gradient_repeat_max_abs_delta"] = max(
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for v in values
            for a, b in zip(jax.tree.leaves(v), jax.tree.leaves(values[0]))
        )
    report["gradient_max_abs_error"] = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(
            jax.tree.leaves(gradients["compact"]),
            jax.tree.leaves(gradients["heuristic"]),
        )
    )
    report["gradients_agree"] = all(
        np.allclose(a, b, rtol=1e-11, atol=1e-9)
        for a, b in zip(
            jax.tree.leaves(gradients["compact"]),
            jax.tree.leaves(gradients["heuristic"]),
        )
    )
    report["fingerprint_gradients_agree"] = all(
        np.allclose(a, b, rtol=1e-11, atol=1e-9)
        for a, b in zip(
            jax.tree.leaves(gradients["compact"]),
            jax.tree.leaves(gradients["fingerprint"]),
        )
    )
    report["fingerprint_gradient_max_abs_error"] = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(
            jax.tree.leaves(gradients["compact"]),
            jax.tree.leaves(gradients["fingerprint"]),
        )
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert all(
        r["fp64_agreement"] and r["distinct_outputs"] == 1
        for r in report["methods"].values()
    )
    assert report["gradients_agree"] and report["fingerprint_gradients_agree"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    benchmark(parser.parse_args())
