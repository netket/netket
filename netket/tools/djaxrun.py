"""
This is a script to run a distributed job using JAX on a single node.
Useful for testing with pytest under jax distributed.

It works by pretending it's running under SLURM, setting up the environment variables
and launching the command in multiple processes.
"""

import argparse
import os
import random
import socket
import sys
import threading
import re
import subprocess
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.console import Group


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def check_hostname_resolvable():
    """
    Warns if the hostname of this machine cannot be resolved to an address.

    jax binds gloo's TCP device (used for cpu collectives in multi-process runs)
    to the address of the local hostname, so if the hostname is not resolvable
    every process crashes at jax startup with `Unable to find address for:
    <hostname>`. This happens on macOS, where the hostname is often only
    resolvable through mDNS, as `<hostname>.local`.
    """
    hostname = socket.gethostname()
    try:
        socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        print(
            f"[djaxrun] WARNING: the hostname of this machine, '{hostname}', cannot be "
            "resolved to an ip address.\n"
            "  jax will most likely fail to initialize its cpu collectives with\n"
            f"  `Unable to find address for: {hostname}`.\n"
            "  To fix this, make the hostname resolvable, for example by running\n"
            f"    sudo scutil --set HostName {hostname}.local     # macOS\n"
            f"    echo '127.0.0.1 {hostname}' | sudo tee -a /etc/hosts   # anywhere",
            file=sys.stderr,
        )


def render_panels(buffers):
    panels = []
    for rank, content in enumerate(buffers):
        text = Text.from_ansi(content.strip()[-1000:])
        panels.append(Panel(text, title=f"[p{rank}]", border_style="green"))
    return Group(*panels)


def main():
    parser = argparse.ArgumentParser(
        description="Distributed runner with live panels and logs, to run a jax.ditributed job on a single node. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-np", type=int, help="Number of processes to run")
    parser.add_argument(
        "-o", type=str, default="djaxrunlog", help="Base output filename prefix"
    )
    parser.add_argument(
        "-s",
        "--simple",
        action="store_true",
        help="Use simple output mode (no Rich panels)",
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=int,
        default=-1,
        help="Filter output to only show output from the process of this rank (negative values means all ranks)",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")

    args = parser.parse_args()

    # Show help if no arguments or missing required ones
    if args.np is None or not args.command:
        parser.print_help()
        sys.exit(1)

    num_procs = args.np
    command = args.command
    outprefix = args.o
    simple_output = args.simple
    filter_output = args.filter

    port = find_free_port()
    job_id = random.randint(100000, 999999)
    node_list = ",".join(["127.0.0.1"] * num_procs)

    buffers = ["" for _ in range(num_procs)]
    threads = []
    procs = []
    log_files = []
    ansi_escape = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

    check_hostname_resolvable()

    def start_proc(rank):
        env = os.environ.copy()
        env.update(
            {
                "JAX_COORDINATOR_ADDRESS": f"127.0.0.1:{port}",
                "SLURM_JOB_ID": str(job_id),
                "SLURM_NTASKS": str(num_procs),
                "SLURM_PROCID": str(rank),
                "SLURM_LOCALID": str(rank),
                "SLURM_STEP_NODELIST": node_list,
                "PYTHONUNBUFFERED": "1",
                "DJAXRUN": "1",
            }
        )

        logfile = open(f"{outprefix}-p{rank}", "w", buffering=1)
        log_files.append(logfile)

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
            text=True,
        )
        procs.append(proc)

        def reader():
            for line in proc.stdout:
                buffers[rank] += line
                cleaned = ansi_escape.sub("", line)
                logfile.write(cleaned)
                if simple_output and (filter_output < 0 or rank == filter_output):
                    print(f"[p{rank}] {line.strip()}")

            proc.stdout.close()
            logfile.close()

        t = threading.Thread(target=reader)
        t.daemon = True
        t.start()
        threads.append(t)

    for rank in range(num_procs):
        start_proc(rank)

    try:
        if simple_output:
            for t in threads:
                t.join()
        else:
            with Live(
                render_panels(buffers), refresh_per_second=10, screen=False
            ) as live:
                while any(t.is_alive() for t in threads):
                    live.update(render_panels(buffers))
                    for t in threads:
                        t.join(timeout=0.1)
    except KeyboardInterrupt:
        print("\n[!] Ctrl-C detected, terminating processes...", file=sys.stderr)
    finally:
        # First, try to terminate all still-running processes
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        # Now wait for all to exit
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        # Close all log files
        for logfile in log_files:
            if not logfile.closed:
                logfile.close()

        # Check exit codes
        exit_codes = [proc.returncode for proc in procs]
        print(f"Exit codes of the processes where {exit_codes}")
        if any(code != 0 for code in exit_codes):
            sys.exit(1)


if __name__ == "__main__":
    main()
