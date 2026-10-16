# Copyright 2026 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Interleave exact source-kernel baselines and the selected implementation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import flax.serialization
import jax
import jax.numpy as jnp
import netket as nk
from netket.vqs.mc import kernels
import numpy as np

from benchmark_unique import build_case, digest


def load_kernel_module(name, directory):
    filename = directory / "netket/vqs/mc/kernels.py"
    spec = importlib.util.spec_from_file_location(name, filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result, filename


def main(args):
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
    old, paths = {}, {}
    for name, source in (
        ("main", "main"),
        ("pr2293", "pr2293"),
        ("published", "candidate"),
    ):
        old[name], paths[name] = load_kernel_module(
            "baseline_" + name, args.sources / source
        )
    paths["optimized_flat"] = paths["optimized_dedup"] = Path(kernels.__file__)
    methods = dict(
        main=old["main"].local_value_kernel_jax_chunked,
        pr2293=old["pr2293"].local_value_kernel_jax_flattened,
        published=old["published"].local_value_kernel_jax_fingerprint,
        optimized_flat=kernels.local_value_kernel_jax_flattened,
        optimized_dedup=kernels.local_value_kernel_jax_fingerprint,
    )
    revisions = dict(
        main="5e4511b7883d89d83b0aba534b37c3c672a15a42",
        pr2293="edeaf94d2768384e36f9db9480de54b5efbe8866",
        published="d45f0e76c6898879cc7f278af701c03845c2a64e",
        optimized_flat="fixed-blocks-prepacked-empty-block-bypass",
        optimized_dedup="fixed-blocks-prepacked-empty-block-bypass",
    )
    runs, values, reports = {}, {}, {}
    for name, kernel in methods.items():
        report = dict(
            **metadata,
            variant=name,
            revision=revisions[name],
            kernel=kernel.__name__,
            kernel_source_sha256=hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            jax_version=jax.__version__,
            device=str(jax.devices()[0]),
            chunk_size=128,
            seconds=[],
            output_hashes=[],
            public_api_agrees=False,
        )
        start = time.perf_counter()
        run = (
            jax.jit(lambda v, x, H: kernel(vs._apply_fun, v, x, H, chunk_size=128))
            .lower(vs.variables, x, H)
            .compile()
        )
        report["compile_s"] = time.perf_counter() - start
        memory = run.memory_analysis()
        report["memory"] = {
            k: getattr(memory, k)
            for k in (
                "argument_size_in_bytes",
                "temp_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
            )
        }
        for _ in range(10):
            value = jax.block_until_ready(run(vs.variables, x, H))
        runs[name], values[name], reports[name] = run, np.asarray(value), report
        np.testing.assert_allclose(value, values["main"], rtol=1e-12, atol=1e-10)
        if name in old:
            # These baselines were also measured in separate processes via the
            # real MCState dispatcher. Require the same saved input hashes and
            # numerical result here, before accepting their interleaved timing.
            prior = json.loads(
                (args.reference_results / f"{args.case}-{name}.json").read_text()
            )
            for key in (
                "variables_sha256",
                "samples_sha256",
                "connectivity_sha256",
                "kernel_source_sha256",
                "kernel",
            ):
                assert prior[key] == report[key], (name, key)
            assert prior["public_api_agrees"]
            np.testing.assert_allclose(
                value,
                np.load(args.reference_results / f"{args.case}-{name}.npy"),
                rtol=1e-12,
                atol=1e-10,
            )
            report["public_api_agrees"] = True
            report["public_api_check"] = (
                "Matches the verified separate-process baseline on identical saved inputs"
            )
        else:
            nk.config.netket_experimental_unique_kernel = name == "optimized_dedup"
            nk.config.netket_experimental_flattened_kernel = name == "optimized_flat"
            assert nk.vqs.get_local_kernel(vs, H, 128) is kernel
            vs._samples = x.reshape(vs.sampler.n_chains, -1, vs.hilbert.size)
            public = np.asarray(vs.local_estimators(H, chunk_size=128).data).reshape(-1)
            np.testing.assert_allclose(value, public, rtol=1e-12, atol=1e-10)
            report["public_api_agrees"] = True
            report["public_api_check"] = (
                "Current dispatcher and local_estimators checked directly"
            )
        print(args.case, name, "compiled", round(report["compile_s"], 3), flush=True)
    names = list(methods)
    for repeat in range(21):
        for name in names[repeat % 5 :] + names[: repeat % 5]:
            start = time.perf_counter()
            value = jax.block_until_ready(runs[name](vs.variables, x, H))
            reports[name]["seconds"].append(time.perf_counter() - start)
            reports[name]["output_hashes"].append(digest(value))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        report.update(
            median_ms=1e3 * float(np.median(report["seconds"])),
            p10_ms=1e3 * float(np.percentile(report["seconds"], 10)),
            p90_ms=1e3 * float(np.percentile(report["seconds"], 90)),
            distinct_outputs=len(set(report["output_hashes"])),
        )
        assert report["distinct_outputs"] == 1
        (args.output / f"{args.case}-{name}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        np.save(args.output / f"{args.case}-{name}.npy", values[name])
        print(args.case, name, f"{report['median_ms']:.6f} ms", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.config.jax_enable_x64 and jax.device_count() == 1
    main(args)
