"""GPU selection utilities for multi-GPU training."""

import os
import subprocess
import numpy as np
from typing import List, Optional


def query_gpu_memory() -> List[dict]:
    """Query GPU memory usage using nvidia-smi.

    Returns:
        List of dicts with 'index', 'name', 'memory_used', 'memory_total', 'memory_free'
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': float(parts[2]),
                    'memory_total': float(parts[3]),
                    'memory_free': float(parts[4]),
                })

        return gpus

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not query nvidia-smi: {e}")
        return []


def select_gpus(n_gpus: int = 2, exclude: Optional[List[int]] = None) -> List[int]:
    """Select n_gpus with lowest memory usage.

    Args:
        n_gpus: Number of GPUs to select
        exclude: List of GPU indices to exclude from selection

    Returns:
        List of selected GPU indices, sorted by memory usage (lowest first)
    """
    if n_gpus <= 0:
        return []

    if exclude is None:
        exclude = []

    gpus = query_gpu_memory()

    if not gpus:
        print("Warning: Could not query GPU memory; leaving CUDA_VISIBLE_DEVICES unchanged")
        return []

    # Filter out excluded GPUs
    available_gpus = [g for g in gpus if g['index'] not in exclude]

    if len(available_gpus) < n_gpus:
        print(f"Warning: Only {len(available_gpus)} GPUs available, requested {n_gpus}")
        n_gpus = len(available_gpus)

    # Sort by memory usage (lowest first)
    available_gpus.sort(key=lambda g: g['memory_used'])

    # Select top n_gpus
    selected = [g['index'] for g in available_gpus[:n_gpus]]

    # Print selection info
    print(f"Selected {len(selected)} GPUs (by memory usage):")
    for idx in selected:
        gpu = next(g for g in available_gpus if g['index'] == idx)
        print(f"  GPU {idx}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_used']:.0f}MB / {gpu['memory_total']:.0f}MB "
              f"({gpu['memory_free']:.0f}MB free)")

    return selected


def set_cuda_visible_devices(gpu_ids: List[int]):
    """Set CUDA_VISIBLE_DEVICES environment variable.

    Args:
        gpu_ids: List of GPU indices to make visible
    """
    visible = ','.join(str(i) for i in gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = visible
    print(f"Set CUDA_VISIBLE_DEVICES={visible}")


def get_available_gpus() -> List[int]:
    """Get list of currently available GPUs.

    Returns:
        List of GPU indices visible to CUDA
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    if not visible:
        # All GPUs available
        gpus = query_gpu_memory()
        return [g['index'] for g in gpus]

    # Parse CUDA_VISIBLE_DEVICES
    return [int(i) for i in visible.split(',') if i]


def print_gpu_status():
    """Print detailed GPU status."""
    gpus = query_gpu_memory()

    if not gpus:
        print("No GPUs detected")
        return

    print("=" * 70)
    print("GPU Status")
    print("=" * 70)
    print(f"{'Index':<8} {'Name':<30} {'Used':<12} {'Total':<12} {'Free':<12}")
    print("-" * 70)

    for gpu in gpus:
        print(f"{gpu['index']:<8} {gpu['name']:<30} "
              f"{gpu['memory_used']:>10.0f}MB "
              f"{gpu['memory_total']:>10.0f}MB "
              f"{gpu['memory_free']:>10.0f}MB")

    print("=" * 70)


if __name__ == '__main__':
    # Test the GPU selection
    print_gpu_status()
    print()

    # Select 2 GPUs with lowest memory usage
    selected = select_gpus(n_gpus=2)
    print(f"\nSelected GPUs: {selected}")

    # Set CUDA_VISIBLE_DEVICES
    set_cuda_visible_devices(selected)

    print(f"\nAvailable GPUs after setting: {get_available_gpus()}")
