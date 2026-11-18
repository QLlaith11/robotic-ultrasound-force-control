import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rusim.patient import PatientMotion


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def grouped_metric(data, metric):
    groups={}
    for row in data:
        groups.setdefault((row["organ"], row["controller"]), []).append(float(row[metric]))
    keys=sorted(groups)
    return keys, [np.mean(groups[key]) for key in keys]


def main():
    output=Path("results/figures")
    output.mkdir(parents=True, exist_ok=True)
    metrics=rows("results/metrics.csv")
    keys_a, rmse=grouped_metric(metrics, "rmse")
    _, mae=grouped_metric(metrics, "mae")
    _, p95=grouped_metric(metrics, "p95_abs_error")
    _, settling=grouped_metric(metrics, "settling_time")
    _, exposure=grouped_metric(metrics, "pct_above_15")
    labels=[f"{organ}\n{controller}" for organ, controller in keys_a]
    fig, axes=plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels=[("RMSE (N)",rmse),("MAE (N)",mae),("95th percentile error (N)",p95),
            ("Settling time (s)",settling),("Time above 15 N (%)",exposure)]
    for ax,(title,values) in zip(axes.flat, panels):
        ax.bar(np.arange(len(values)), values, color="#2878b5")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=.25)
    axes.flat[-1].axis("off")
    fig.suptitle("Controller comparison across organs and held-out seeds")
    fig.savefig(output/"controller_dashboard.png", dpi=180)
    fig.savefig(output/"controller_dashboard.svg")
    plt.close(fig)

    times=np.linspace(0, 20, 2001)
    fig, ax=plt.subplots(figsize=(10, 4))
    for organ,color in (("liver","#2878b5"),("prostate","#d95f02")):
        motion=PatientMotion(organ, 0)
        displacement=np.array([motion.breathing_displacement(t) for t in times])*1000
        ax.plot(times, displacement, color=color, label=f"{organ} breathing component")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Surface displacement (mm)")
    ax.set_title("Respiratory tissue-surface motion used by the benchmark")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"respiratory_motion.png", dpi=180)
    fig.savefig(output/"respiratory_motion.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
