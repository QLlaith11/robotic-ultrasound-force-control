from rusim.simulation import benchmark
from rusim.report import make_figure
if __name__ == "__main__":
    benchmark("results", range(10), 6.0)
    make_figure()
