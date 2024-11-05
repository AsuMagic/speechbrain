"""Wrapper to handle PyTorch profiling and benchmarking.

Author:
    * Titouan Parcollet 2024
"""

import torch
import os
import time


class ProfiledIterable:
    def __init__(self, iterable, warn_threshold: float = 0.1, for_steps: int = 5):
        self.iterable = iterable
        self.iterator = iter(iterable)
        self.warn_threshold = warn_threshold
        self.for_steps = for_steps

        self.last_times = []

    def __iter__(self):
        return self

    def __next__(self):
        start_time = time.time()
        element = next(self.iterator)
        end_time = time.time()

        self.last_times.append(end_time - start_time)

        if len(self.last_times) > self.for_steps:
            self.last_times = self.last_times[1:]

        time_last_steps = sum(self.last_times)

        if time_last_steps > self.warn_threshold:
            print(f"!!! Data loading slow: took {time_last_steps}s total for the past {self.for_steps} iterations; consider increasing workers or verifying IO bottlenecking issues")
        else:
            print(f"    ok {end_time - start_time}")

        return element

    def __len__(self):
        return len(self.iterable)


def profiled_iterable(iterable, warn_threshold = 0.1):
    iterator = iter(iterable)

    while True:
        try:
            start_time = time.time()
            element = next(iterator)
            end_time = time.time()

            yield element

            time_diff = end_time - start_time
            if time_diff > warn_threshold:
                print("!", time_diff)
            else:
                print(" ", time_diff)
        except StopIteration:
            return

from torch import profiler

def default_trace_handler(prof, logdir):
    prof.export_stacks(os.path.join(logdir, "cpu_stacks.txt"), metric="self_cpu_time_total")
    prof.export_stacks(os.path.join(logdir, "gpu_stacks.txt"), metric="self_cuda_time_total")
    prof.export_chrome_trace(os.path.join(logdir, "trace.json"))


def prepare_profiler(
    profile_warmup=5, profile_steps=5, logdir="tensorboard_logs"
):
    """Wrapper to create a PyTorch profiler to benchmark training of speechbrain.core.Brain instances.
    See ``torch.profiler.profile`` documentation for details (brief summary below).

    Arguments
    ---------
    profile_warmup: int
        Number of warmup step before starting to log.
    profile_steps: int
        Number of steps to log after warmup.
    logdir: str
        Path to the output folder of the logs.

    Returns
    -------
    profiler
    """
    logdir = os.path.join(logdir, "profiler_logs")

    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=0, warmup=profile_warmup, active=profile_steps, repeat=1
        ),
        on_trace_ready=lambda prof: default_trace_handler(prof, logdir),
        # record_shapes=True,
        with_stack=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    )
