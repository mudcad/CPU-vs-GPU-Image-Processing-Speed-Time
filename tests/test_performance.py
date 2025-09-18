from mypackage.image_processor import benchmark_and_plot


if __name__ == "__main__":
    sizes = [256, 512, 1024, 2048]
    benchmark_and_plot(sizes, channels=3, trials=5, seed=42)
