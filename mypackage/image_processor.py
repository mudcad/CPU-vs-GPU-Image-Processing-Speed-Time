import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def benchmark_and_plot(sizes, channels=3, trials=5, seed=42):
    """
    Benchmark CPU vs GPU performance for simple image operations
    on random images of different sizes.
    """

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ Using device: {device}")

    cpu_times = []
    gpu_times = []

    for size in sizes:
        print(f"\n--- Benchmarking {size}x{size} ---")

        # Generate random image
        img = torch.rand((channels, size, size), dtype=torch.float32)

        # ----- CPU benchmark -----
        start = time.time()
        for _ in range(trials):
            _ = img * 1.5 + 2.0
        end = time.time()
        cpu_time = (end - start) / trials
        cpu_times.append(cpu_time)
        print(f"CPU: {cpu_time:.6f} s")

        # ----- GPU benchmark -----
        if device.type == "cuda":
            img_gpu = img.to(device)

            # Warmup
            for _ in range(3):
                _ = img_gpu * 1.5 + 2.0
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(trials):
                _ = img_gpu * 1.5 + 2.0
            torch.cuda.synchronize()
            end = time.time()

            gpu_time = (end - start) / trials
            gpu_times.append(gpu_time)
            print(f"GPU: {gpu_time:.6f} s")
        else:
            gpu_times.append(None)
            print("⚠️ GPU not available.")

    # ----- Plot results -----
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, cpu_times, marker="o", label="CPU")
    if any(t is not None for t in gpu_times):
        plt.plot(sizes, [t if t is not None else float("nan") for t in gpu_times],
                 marker="o", label="GPU")

    plt.xlabel("Image size (pixels)")
    plt.ylabel("Average time per trial (s)")
    plt.title("CPU vs GPU Image Processing Performance")
    plt.legend()
    plt.grid(True)
    plt.show()
