import argparse
from .simulation import benchmark
from .experiments import rigorous, rigorous_checkpointed, stress_test
from .report import make_figure, make_metric_figures
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="results")
    p.add_argument("--fast",action="store_true"); p.add_argument("--rigorous",action="store_true")
    p.add_argument("--checkpointed",action="store_true"); p.add_argument("--stress",action="store_true")
    p.add_argument("--backend",choices=("mujoco","surrogate"),default="mujoco")
    p.add_argument("--seeds",type=int,default=10); p.add_argument("--duration",type=float,default=None)
    a=p.parse_args()
    if a.stress:
        rows=stress_test(f"{a.output}/stress_test.csv",a.backend,range(min(a.seeds,3)),a.duration or 8.,2.5)
        print(f"wrote {len(rows)} stress-test rows to {a.output}/stress_test.csv")
        return
    if a.rigorous:
        runner = rigorous_checkpointed if a.checkpointed else rigorous
        runner(a.output,a.backend,range(a.seeds),a.duration or 60.,min(a.duration or 10.,10.))
    else: benchmark(a.output,range(10),6. if a.fast else 60.,a.backend)
    make_figure(f"{a.output}/metrics.csv", f"{a.output}/figures/rmse.svg")
    make_metric_figures(f"{a.output}/metrics.csv", f"{a.output}/figures")
    print(f"wrote {a.output}/metrics.csv and {a.output}/figures/rmse.svg")
if __name__=="__main__": main()
