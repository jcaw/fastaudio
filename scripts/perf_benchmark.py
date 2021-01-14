#!python

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import sys
import time
import torch
from copy import copy
from fastai.vision.all import RandTransform, Transform
from pathlib import Path
from tqdm import tqdm
from warnings import warn

from fastaudio.all import (
    AddNoise,
    AddNoiseGPU,
    AudioSpectrogram,
    AudioTensor,
    ChangeVolume,
    ChangeVolumeGPU,
    Delta,
    DeltaGPU,
    MaskFreq,
    MaskFreqGPU,
    MaskTime,
    MaskTimeGPU,
    NoiseColor,
    SignalCutout,
    SignalCutoutGPU,
    SignalLoss,
    SignalLossGPU,
    TfmResize,
    TfmResizeGPU,
    auto_batch,
)

BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
SCRIPTS_DIR = BASE_DIR / "scripts"
# Can just manually change this if running on colab
REPEATS = 1000
if __name__ == "__main__" and len(sys.argv) > 1:
    OUTPUT_DIR = Path(sys.argv[1]) / "perf_output"
    print(f'Output dir set to "{OUTPUT_DIR}"')
    if len(sys.argv) > 2:
        REPEATS = int(sys.argv[2])
        print(f"Repeats set to {REPEATS}")
else:
    OUTPUT_DIR = SCRIPTS_DIR / "perf_output"


class Timer(object):
    """
    timer: A class used to measure the execution time of a block of code that is
    inside a "with" statement.

    Example:

    ```
    with timer("Count to 500000"):
        x = 0
        for i in range(500000):
            x += 1
        print(x)
    ```

    Will output:
    500000
    Count to 500000: 0.04 s

    Warning: The time resolution used here may be limited to 1 ms
    """

    def __init__(self, device=None, description="Execution time", verbose=False):
        self.description = description
        self.verbose = verbose
        self.execution_time = None
        self.cuda = device.type == "cuda"

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if self.cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            # Have to convert from ms to seconds
            self.execution_time = self.start_event.elapsed_time(self.end_event) / 1000
        else:
            self.execution_time = time.time() - self.t
        if self.verbose:
            print("{}: {:.3f} s".format(self.description, self.execution_time))


def measure_execution_time(
    transform, tensor_type, batch_size, duration, device_name, device, repeats
):
    transform_name = transform.__class__.__name__
    # Just fix these for now, makes the structure easier.
    NUM_CHANNELS = 2
    SAMPLE_RATE = 16000
    perf_objects = []
    for i in range(repeats):
        perf_obj = {
            "metrics": {},
            "params": {
                "transform": transform_name,
                "batch_size": batch_size,
                "duration": duration,
                "device_name": device_name,
            },
        }
        batch_dim = (batch_size,) if batch_size else tuple()
        if tensor_type == AudioTensor:
            n_samples = round(duration * SAMPLE_RATE)
            shape = batch_dim + (NUM_CHANNELS, n_samples)
            tensor = AudioTensor(
                torch.rand(shape, dtype=torch.float32, device=device), sr=SAMPLE_RATE
            )
        elif tensor_type == AudioSpectrogram:
            # Just use a sensible constant for now.
            N_FREQ_BANDS = 200
            SPECGRAM_STEP_MS = 10
            # TODO: Maybe extract this computation (but it's probably negligible)
            n_steps = round(duration * SAMPLE_RATE // SPECGRAM_STEP_MS)
            shape = batch_dim + (NUM_CHANNELS, n_steps, N_FREQ_BANDS)
            tensor = AudioSpectrogram(
                torch.rand(shape, dtype=torch.float32, device=device) - 1
            )
        else:
            raise ValueError("Unrecognized tensor type")
        t = Timer(device=device)
        if isinstance(transform, RandTransform):
            with t:
                transform(tensor, split_idx=0)
        else:
            with t:
                transform(tensor)

        perf_obj["metrics"]["execution_time"] = t.execution_time
        perf_objects.append(perf_obj)
    return perf_objects


def run_benchmark(
    subdir,
    tensor_type,
    transforms,
    batch_sizes=[32],
    durations=[1],
    device_names=["cpu", "cuda"],
    repeats=REPEATS,
):
    """
    For each transformation, apply it to an example sound and write the transformed sounds to
    an output folder. Also crudely measure and print execution time.
    """
    print("-" * 50)
    print(f"Running benchmark: {subdir}")

    np.random.seed(42)
    random.seed(42)

    output_dir = OUTPUT_DIR / subdir
    os.makedirs(output_dir, exist_ok=True)

    device_names = copy(device_names)
    if not torch.cuda.is_available():
        warn("CUDA is not available - removing CUDA from devices.")
        device_names.remove("cuda")

    devices = {device_name: torch.device(device_name) for device_name in device_names}

    perf_objects = []

    for device_name, device in devices.items():
        for batch_size in tqdm(batch_sizes, desc=device_name):
            for duration in durations:
                for transform in transforms:
                    try:
                        perf_objects += measure_execution_time(
                            transform,
                            tensor_type,
                            batch_size,
                            duration,
                            device_name,
                            device,
                            repeats,
                        )
                    except Exception as e:
                        # Don't crash out on error
                        print(
                            f"Error timing bs {batch_size}, time {duration}, device {device}"
                            f"\n  `{transform}`\n"
                            # Hack to get around flake8 hook
                            + str(e),
                            file=sys.stderr,
                        )

    params_to_group_by = ["batch_size", "duration", "device_name"]
    for group_by_param in tqdm(params_to_group_by, desc="Making plots"):
        param_values = []
        metric_values = []
        transform_names = []
        for perf_obj in perf_objects:
            param_values.append(perf_obj["params"][group_by_param])
            metric_values.append(perf_obj["metrics"]["execution_time"])
            transform_names.append(perf_obj["params"]["transform"])

        df = pd.DataFrame(
            {
                group_by_param: param_values,
                "exec_time": metric_values,
                "transform": transform_names,
            }
        )

        violin_plot_file_path = str(output_dir / "{}_plot.png".format(group_by_param))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("execution time grouped by {}".format(group_by_param))
        g = sns.boxplot(
            x=group_by_param,
            y="exec_time",
            data=df,
            ax=ax,
            hue="transform",
            linewidth=1.0,
            fliersize=2.0,
        )
        g.set_yscale("log")
        fig.tight_layout()
        plt.savefig(violin_plot_file_path)
        plt.close(fig)


def basic_benchmark():
    # Old vs. new on CPU
    run_benchmark(
        subdir="AudioTensor_old_vs_new_cpu",
        tensor_type=AudioTensor,
        transforms=[
            AddNoise(color=NoiseColor.Pink),
            AddNoiseGPU(p=1.0, color=NoiseColor.Pink),
            ChangeVolume(p=1.0),
            ChangeVolumeGPU(p=1.0),
            SignalCutout(p=1.0),
            SignalCutoutGPU(p=1.0),
            SignalLoss(p=1.0),
            SignalLossGPU(p=1.0),
        ],
        # The old transforms can only cope with items, not batches, so
        # pass in items.
        batch_sizes=[None],
        device_names=["cpu"],
    )

    run_benchmark(
        subdir="AudioSpectrogram_old_vs_new_cpu",
        tensor_type=AudioSpectrogram,
        transforms=[
            TfmResize(size=533, interp_mode="bilinear"),
            TfmResizeGPU(size=533, interp_mode="bilinear"),
            Delta(width=9),
            DeltaGPU(width=9),
            MaskTime(num_masks=2),
            MaskTimeGPU(num_masks=2),
            MaskFreq(num_masks=2),
            MaskFreqGPU(num_masks=2),
        ],
        # The old transforms can only cope with items, not batches, so
        # pass in items.
        batch_sizes=[None],
        device_names=["cpu"],
    )

    # Where vs. scatter for ChangeVolume (CUDA only)

    new_signal_tfms = [
        AddNoiseGPU(p=1.0, color=NoiseColor.Pink),
        ChangeVolumeGPU(p=1.0),
        SignalCutoutGPU(p=1.0),
        SignalLossGPU(p=1.0),
    ]
    new_spec_tfms = [
        TfmResizeGPU(size=533, interp_mode="bilinear"),
        DeltaGPU(width=9, mode="reflect"),
        MaskTimeGPU(num_masks=2),
        MaskFreqGPU(num_masks=2),
    ]

    for tensor_type, transforms in [
        (AudioTensor, new_signal_tfms),
        (AudioSpectrogram, new_spec_tfms),
    ]:
        # Benchmark differences between batch sizes in general
        prefix = tensor_type.__name__
        run_benchmark(
            subdir=f"{prefix}_gpu_by_batch_size",
            tensor_type=tensor_type,
            transforms=transforms,
            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            device_names=["cuda"],
        )
        run_benchmark(
            subdir=f"{prefix}_gpu_by_length",
            tensor_type=tensor_type,
            transforms=transforms,
            # Use a smaller batch size & vary the duration
            batch_sizes=[16],
            durations=[0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64],
            device_names=["cuda"],
        )

        # Benchmark new transforms on CPU vs. GPU, with a few batch sizes.
        run_benchmark(
            subdir=f"{prefix}_cpu_vs_gpu_bs_32",
            tensor_type=tensor_type,
            transforms=transforms,
            batch_sizes=[32],
        )
        run_benchmark(
            subdir=f"{prefix}_cpu_vs_gpu_bs_64",
            tensor_type=tensor_type,
            transforms=transforms,
            batch_sizes=[64],
            # This one is slower, so do less repeats.
            repeats=max(REPEATS // 4, 1),
        )

    # Overhead of the autobatch wrapper
    class CleanTransform(Transform):
        def encodes(self, ai: AudioTensor):
            return ai

    class AutoBatchTransform(Transform):
        @auto_batch(2)
        def encodes(self, ai: AudioTensor):
            return ai

    run_benchmark(
        subdir="autobatch_wrapper",
        tensor_type=AudioTensor,
        transforms=[
            CleanTransform(),
            AutoBatchTransform(),
            AutoBatchTransform(),
            CleanTransform(),
        ],
        # This one is really fast so run a few more times for a
        # better estimate.
        repeats=REPEATS * 5,
    )


if __name__ == "__main__":
    basic_benchmark()
