import subprocess
import os
from datetime import timedelta, datetime

import jax

from netket.utils import struct

from netket._src.callbacks.base import AbstractCallback


def get_time_left(job_id):
    """Queries the remaining time for the current Slurm job using the SLURM_JOB_ID environment variable."""
    # Retrieve the job ID from the environment variable
    try:
        # Run the squeue command to fetch the remaining time
        result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-O", "TimeLeft"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Extract the time left from the command's output
        time_left_str = result.stdout.strip()
        if not time_left_str or time_left_str == "UNLIMITED":
            # print(f"Job {job_id} has unlimited or no time limit.")
            return None

        # Parse the time left dynamically
        if "-" in time_left_str:
            days, time_str = time_left_str.split("-")
            days = int(days)
        else:
            days = 0
            time_str = time_left_str

        # Split the time string and pad missing fields with zeros
        time_parts = list(map(int, time_str.split(":")))
        while len(time_parts) < 3:  # Ensure [hours, minutes, seconds]
            time_parts.insert(0, 0)

        hours, minutes, seconds = time_parts

        # Calculate total seconds and convert to timedelta
        total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
        return timedelta(seconds=total_seconds)

    except subprocess.TimeoutExpired:
        print("squeue command timed out.")
    except subprocess.CalledProcessError as e:
        print(f"Error querying squeue: {e}")
    except ValueError as e:
        print(f"Error parsing time left: {e}")

    return None


def is_requeueable(jobid):
    """
    Check if a Slurm job is requeueable.

    Args:
        jobid (str): The job ID to check.

    Returns:
        bool: True if the job is requeueable, False otherwise.
    """
    try:
        # Run scontrol to get job details
        result = subprocess.run(
            ["scontrol", "show", "job", jobid],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse the output for the Requeue field
        for line in result.stdout.splitlines():
            if "Requeue=" in line:
                requeue_value = line.split("Requeue=")[1].split()[0]
                return requeue_value == "1"

        # If the Requeue field is not found
        raise RuntimeError(f"Could not determine Requeue status for job {jobid}.")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error while running scontrol: {e.stderr.strip()}")


class AutoSlurmRequeue(AbstractCallback):
    """
    A callback that automatically requeues a Slurm job if it is about to run out
    of time.

    This callback should be used together with a form of checkpointing to ensure
    that the job can be requeued without losing progress.
    """

    before: timedelta = struct.field(pytree_node=False, serialize=False)
    max_requeue_count: int = struct.field(pytree_node=False, serialize=False)
    _time_to_requeue: timedelta | None = struct.field(
        pytree_node=False, serialize=False, default=None
    )
    _enabled: bool = struct.field(pytree_node=False, serialize=False, default=True)

    def __init__(
        self, before: timedelta = timedelta(minutes=5), max_requeue_count: int = 3
    ):
        """
        Initialize the auto-requeue callback.

        Args:
            before: The time before the job ends to check for requeueing (default: 5 minute).
                This should be a timedelta object or a number of seconds, and it should be at least
                as long as the time it takes an iteration to run.
            max_requeue_count: Maximum number of times the job should be requeued.
        """
        if isinstance(before, (int)):
            before = timedelta(seconds=before)
        if not isinstance(before, timedelta):
            raise TypeError(
                f"Expected before to be a timedelta or an int (seconds), but got {type(before)}"
            )
        if not isinstance(max_requeue_count, int) or max_requeue_count < 0:
            raise ValueError("max_requeue_count must be a non-negative integer.")

        self.before = before
        self.max_requeue_count = max_requeue_count

    def on_run_start(self, step, driver):
        jobid = os.getenv("SLURM_JOB_ID")
        requeue_count = int(os.getenv("SLURM_RESTART_COUNT", "0"))

        if jobid is None:
            print("SLURM_JOB_ID not found. Skipping auto-requeue check.")
            self._enabled = False
            return

        if requeue_count >= self.max_requeue_count:
            print(
                f"Job has been requeued {requeue_count} times, exceeding the limit of {self.max_requeue_count}. Disabling auto-requeue."
            )
            self._enabled = False
            return

        if not is_requeueable(jobid):
            raise RuntimeError(
                "Job is not requeable. Use `--requeue` option when scheduling the job \n"
                "for example use `sbatch --requeue file` or add #SLURM --requeue at the top"
                "of the file."
            )

        time_left = get_time_left(jobid)

        if time_left is None:
            print("No time limit found. Skipping auto-requeue check.")
            self._enabled = False
            return

        self._time_to_requeue = datetime.now() + time_left - self.before
        self._enabled = True

    def on_step_end(self, step, log_data, driver):
        if not self._enabled:
            return

        if datetime.now() > self._time_to_requeue:
            print("Reached requeue time. Requeueing job...")
            if jax.process_index() == 0:
                subprocess.run(["scontrol", "requeue", os.getenv("SLURM_JOB_ID")])
            self._enabled = False
