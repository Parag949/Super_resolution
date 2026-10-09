import os
import glob
import time
import warnings
import numpy as np
import rasterio
import multiprocessing as mp
from rasterio.errors import NotGeoreferencedWarning

# Suppress rasterio unreferenced coordinate warnings
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def _process_single_image(args):
    """
    Worker function executed in parallel across CPU cores.
    Reads specified bands and returns (min_per_channel, max_per_channel, mean_per_channel).
    """
    file_path, bands = args
    try:
        with rasterio.open(file_path) as src:
            # Read shape: (C, H, W) where C is len(bands)
            data = src.read(bands).astype(np.float32)
            
            c_min = data.min(axis=(1, 2))
            c_max = data.max(axis=(1, 2))
            c_mean = data.mean(axis=(1, 2))
            
            return (c_min, c_max, c_mean)
    except Exception as e:
        return None


def compute_parallel_stats(file_paths, bands, dataset_name="Dataset", num_workers=None):
    """
    Computes global and per-channel min, max, and mean across all images in parallel.
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 2)

    total_files = len(file_paths)
    print(f"\n=================================================================")
    print(f"🚀 Processing {dataset_name}: {total_files} files using {num_workers} parallel workers")
    print(f"=================================================================")

    start_time = time.time()
    task_args = [(f, bands) for f in file_paths]

    # Use 'fork' context for seamless speed in Linux & Jupyter
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=num_workers) as pool:
        # chunksize=32 batches the work to minimize inter-process communication overhead
        results = pool.map(_process_single_image, task_args, chunksize=32)

    # Filter out any failed reads
    valid_results = [r for r in results if r is not None]
    elapsed = time.time() - start_time

    if not valid_results:
        print(f"⚠️ No valid results found for {dataset_name}!")
        return None

    # Stack results: shape (N, C)
    mins = np.array([r[0] for r in valid_results])
    maxs = np.array([r[1] for r in valid_results])
    means = np.array([r[2] for r in valid_results])

    # Global and per-channel aggregates
    global_min = float(mins.min())
    global_max = float(maxs.max())

    per_channel_min = mins.min(axis=0)
    per_channel_max = maxs.max(axis=0)
    per_channel_mean = means.mean(axis=0)

    print(f"✅ Finished in {elapsed:.2f} seconds ({len(valid_results)}/{total_files} files read)")
    print(f"📊 OVERALL GLOBAL MIN: {global_min:.4f}")
    print(f"📊 OVERALL GLOBAL MAX: {global_max:.4f}")
    print("\n--- Per-Channel Breakdown (Red, Green, Blue) ---")
    channel_names = ["Red", "Green", "Blue"]
    for i in range(len(bands)):
        name = channel_names[i] if i < len(channel_names) else f"Band {bands[i]}"
        print(f"  [{name}] Min: {per_channel_min[i]:.4f} | Max: {per_channel_max[i]:.4f} | Mean: {per_channel_mean[i]:.4f}")

    return {
        "global_min": global_min,
        "global_max": global_max,
        "channel_min": per_channel_min,
        "channel_max": per_channel_max,
        "channel_mean": per_channel_mean,
    }


if __name__ == "__main__":
    # -------------------------------------------------------------
    # 1. LOW-RESOLUTION (Sentinel-2 L2A) -> Bands 4 (R), 3 (G), 2 (B)
    # -------------------------------------------------------------
    lr_files = sorted(glob.glob("archive_1/lr_dataset/**/*-L2A_data.tiff", recursive=True))
    if lr_files:
        lr_stats = compute_parallel_stats(
            file_paths=lr_files,
            bands=[4, 3, 2],
            dataset_name="Low-Resolution Sentinel-2 (LR)",
        )

    # -------------------------------------------------------------
    # 2. HIGH-RESOLUTION (SPOT Pansharpened) -> Bands 1 (R), 2 (G), 3 (B)
    # -------------------------------------------------------------
    hr_files = sorted(glob.glob("archive_1/hr_dataset/**/*_ps.tiff", recursive=True))
    if hr_files:
        hr_stats = compute_parallel_stats(
            file_paths=hr_files,
            bands=[1, 2, 3],
            dataset_name="High-Resolution SPOT Pansharpened (HR)",
        )
