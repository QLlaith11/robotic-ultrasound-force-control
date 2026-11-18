import csv, json
from pathlib import Path
import numpy as np
from .simulation import run_episode, benchmark, validate_motion, validate_breathing, bringup
from .tissue import validate_tissue_model

def _write(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

def paired_statistics(rows, metric="rmse", a="pi", b="sta"):
    left={(r["organ"],int(r["seed"])):float(r[metric]) for r in rows if r["controller"]==a}
    right={(r["organ"],int(r["seed"])):float(r[metric]) for r in rows if r["controller"]==b}
    keys=sorted(set(left).intersection(right))
    if not keys: raise ValueError("no paired organ/seed observations")
    x=np.array([left[k] for k in keys]); y=np.array([right[k] for k in keys])
    d=y-x; rng=np.random.default_rng(42); boots=[]
    for _ in range(2000): boots.append(np.mean(rng.choice(d,len(d),replace=True)))
    try:
        from scipy.stats import wilcoxon
        p=float(wilcoxon(d).pvalue)
    except Exception: p=float("nan")
    return {"metric":metric,"a":a,"b":b,"n":len(d),"mean_difference":float(d.mean()),
            "bootstrap_low":float(np.quantile(boots,.025)),"bootstrap_high":float(np.quantile(boots,.975)),
            "wilcoxon_p":p,"pairs":[{"organ":k[0],"seed":k[1]} for k in keys]}

def tune(output="results/gains.json", backend="mujoco", tuning_seeds=range(100,105), duration=5., trajectory_difficulty=1.15):
    # Equal-budget held-out tuning; evaluation seeds never appear here.
    candidates={"pi":[(.004,.08),(.006,.15),(.009,.25)],
                "sta":[(.5,1.0),(.8,2.0),(1.2,3.0)],
                "blsmc":[(3.0,1.5,1.0),(4.0,2.0,1.5),(4.5,2.5,2.0)]}
    scores={}
    for name, vals in candidates.items():
        scores[name]=[]
        for val in vals:
            if name=="pi": gains={"kp":val[0],"ki":val[1]}
            elif name=="sta": gains={"k1":val[0],"k2":val[1]}
            else: gains={"barrier_margin":val[0],"barrier_gain":val[1],"switch_gain":val[2]}
            rs=[run_episode("liver",name,s,duration,backend=backend,gains=gains,
                            trajectory_difficulty=trajectory_difficulty)["rmse"] for s in tuning_seeds]
            scores[name].append((float(np.mean(rs)),val))
    best={k:min(v)[1] for k,v in scores.items()}
    selected={"pi":{"kp":best["pi"][0],"ki":best["pi"][1]},
              "sta":{"k1":best["sta"][0],"k2":best["sta"][1]},
              "blsmc":{"barrier_margin":best["blsmc"][0],
                       "barrier_gain":best["blsmc"][1],
                       "switch_gain":best["blsmc"][2]}}
    data={"tuning_seeds":list(tuning_seeds),"evaluation_seeds":list(range(10)),
          "selected":selected,"candidate_scores":scores}
    Path(output).write_text(json.dumps(data,indent=2)); return data

def rigorous(output_dir="results", backend="mujoco", evaluation_seeds=range(10),
             main_duration=60., study_duration=10., trajectory_difficulty=1.15):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    validate_motion(out/"motion_validation.csv",range(40)); validate_breathing(out/"breathing_validation.csv",range(40)); bringup(out/"bringup.csv")
    gain_data=tune(out/"gains.json",backend,range(100,105),min(5.,main_duration),trajectory_difficulty)
    rows=[]
    evaluation_seeds=list(evaluation_seeds)
    for organ in ("liver","prostate"):
        for ctl in ("impedance","pi","sta","blsmc"):
            for seed in evaluation_seeds:
                rows.append(run_episode(organ,ctl,seed,main_duration,backend=backend,
                                        gains=gain_data["selected"].get(ctl),
                                        trajectory_difficulty=trajectory_difficulty))
    _write(out/"metrics.csv",rows)
    stats=[paired_statistics([r for r in rows if r["organ"]==organ],m,a,b)
           for organ in ("liver","prostate") for a,b in
           (("pi","sta"),("pi","blsmc"),("sta","blsmc")) for m in
           ("rmse","mae","error_std","p95_abs_error","overshoot","undershoot",
            "settling_time","median","iqr","pct_above_11","pct_above_15",
            "max_force","effort","time_above_15","alarm_rate","retract_rate",
            "force_guard_rate","hard_limit_violation_rate","max_force_excess",
            "raw_force_demand_rate","raw_max_force_excess")]
    _write(out/"paired_statistics.csv",stats)
    est=[]
    for organ in ("liver","prostate"):
        for e in ("static","dynamic","oracle"):
            for s in range(min(3,len(evaluation_seeds))):             est.append(run_episode(organ,"pi",s,study_duration,e,backend=backend,
                                   trajectory_difficulty=trajectory_difficulty))
    _write(out/"estimation.csv",est)
    safety=[]
    configs=[("pi",False,None),("sta",False,None),("sta",True,None),("sta",True,"bias"),("sta",True,"dropout"),("pi",True,"bias")]
    for ctl,safe,fault in configs:
        for s in range(min(3,len(evaluation_seeds))):         safety.append(run_episode("liver",ctl,s,study_duration,safety=safe,fault=fault,backend=backend,
                                  trajectory_difficulty=trajectory_difficulty))
    _write(out/"safety_faults.csv",safety)
    robust=[]
    for label in ("k_minus50","k_plus50","friction_x2","mass_error_x5","torque_noise_x3"):
        for s in range(min(3,len(evaluation_seeds))):
            perturb={"k_minus50":{"stiffness_scale":.5},
                     "k_plus50":{"stiffness_scale":1.5},
                     "friction_x2":{"friction_scale":2.},
                     "mass_error_x5":{"mass_scale":1.25},
                     "torque_noise_x3":{"torque_noise":3.}}[label]
            r=run_episode("liver","pi",s,study_duration,backend=backend,
                          trajectory_difficulty=trajectory_difficulty,**perturb)
            r["perturbation"]=label; robust.append(r)
    _write(out/"robustness.csv",robust)
    return rows


def rigorous_checkpointed(output_dir="results", backend="mujoco",
                          evaluation_seeds=range(10), main_duration=60.,
                          study_duration=10., trajectory_difficulty=1.15):
    """Run the full study with durable per-episode checkpoints.

    Main benchmark episodes are written as individual JSON files before the
    aggregate CSV is replaced. An interrupted run can therefore be resumed
    without discarding completed 60-second episodes.
    """
    out = Path(output_dir)
    checkpoint = out / ".checkpoint_blsmc_v3"
    checkpoint.mkdir(parents=True, exist_ok=True)
    validate_motion(out / "motion_validation.csv", range(40))
    validate_breathing(out / "breathing_validation.csv", range(40))
    validate_tissue_model(out / "tissue_validation.csv")
    bringup(out / "bringup.csv")
    gain_data = tune(out / "gains.json", backend, range(100, 105),
                     min(5., main_duration), trajectory_difficulty)
    evaluation_seeds = list(evaluation_seeds)
    for organ in ("liver", "prostate"):
        for ctl in ("impedance", "pi", "sta","blsmc"):
            for seed in evaluation_seeds:
                path = checkpoint / f"{organ}_{ctl}_{seed}.json"
                if not path.exists():
                    result = run_episode(
                        organ, ctl, seed, main_duration, backend=backend,
                        gains=gain_data["selected"].get(ctl),
                        trajectory_difficulty=trajectory_difficulty,
                    )
                    path.write_text(json.dumps(result, indent=2))
    rows = [
        json.loads(path.read_text())
        for path in sorted(checkpoint.glob("*.json"))
    ]
    _write(out / "metrics.csv", rows)
    stats = [
        paired_statistics([r for r in rows if r["organ"] == organ], metric, a, b)
        for organ in ("liver", "prostate")
        for a, b in (("pi", "sta"), ("pi", "blsmc"), ("sta", "blsmc"))
        for metric in
        ("rmse","mae","error_std","p95_abs_error","overshoot",
         "undershoot","settling_time","median","iqr","pct_above_11",
         "pct_above_15","max_force","effort","time_above_15",
         "alarm_rate","retract_rate","force_guard_rate",
         "hard_limit_violation_rate","max_force_excess",
         "raw_force_demand_rate","raw_max_force_excess")
    ]
    _write(out / "paired_statistics.csv", stats)
    est = []
    for organ in ("liver", "prostate"):
        for estimator in ("static", "dynamic", "oracle"):
            for seed in evaluation_seeds[:3]:
                est.append(run_episode(
                    organ, "pi", seed, study_duration, estimator,
                    backend=backend, gains=gain_data["selected"]["pi"],
                    trajectory_difficulty=trajectory_difficulty,
                ))
    _write(out / "estimation.csv", est)
    safety = []
    configs = [
        ("pi", False, None), ("sta", False, None), ("sta", True, None),
        ("sta", True, "bias"), ("sta", True, "dropout"),
        ("pi", True, "bias"),
    ]
    for ctl, safe, fault in configs:
        for seed in evaluation_seeds[:3]:
            safety.append(run_episode(
                "liver", ctl, seed, study_duration, safety=safe, fault=fault,
                backend=backend, gains=gain_data["selected"].get(ctl),
                trajectory_difficulty=trajectory_difficulty,
            ))
    _write(out / "safety_faults.csv", safety)
    robust = []
    perturbations = {
        "k_minus50": {"stiffness_scale": .5},
        "k_plus50": {"stiffness_scale": 1.5},
        "friction_x2": {"friction_scale": 2.},
        "mass_error_x5": {"mass_scale": 1.25},
        "torque_noise_x3": {"torque_noise": 3.},
    }
    for label, perturbation in perturbations.items():
        for seed in evaluation_seeds[:3]:
            result = run_episode(
                "liver", "pi", seed, study_duration, backend=backend,
                gains=gain_data["selected"]["pi"],
                trajectory_difficulty=trajectory_difficulty, **perturbation,
            )
            result["perturbation"] = label
            robust.append(result)
    _write(out / "robustness.csv", robust)
    return rows

def stress_test(output="results/stress_test.csv", backend="mujoco",
                seeds=range(3), duration=8., trajectory_difficulty=2.5):
    rows = []
    cases = {
        "rapid_breathing": {"trajectory_difficulty": trajectory_difficulty},
        "soft_tissue": {"trajectory_difficulty": trajectory_difficulty, "stiffness_scale": .5},
        "stiff_tissue": {"trajectory_difficulty": trajectory_difficulty, "stiffness_scale": 2.5},
        "mass_mismatch": {"trajectory_difficulty": trajectory_difficulty, "mass_scale": 2.0},
        "torque_noise": {"trajectory_difficulty": trajectory_difficulty, "torque_noise": 3.},
        "combined_adversary": {"trajectory_difficulty": trajectory_difficulty * 1.2,
                               "stiffness_scale": 2.5, "mass_scale": 2.0,
                               "torque_noise": 3.},
        "sensor_bias": {"trajectory_difficulty": trajectory_difficulty, "fault": "bias"},
        "sensor_dropout": {"trajectory_difficulty": trajectory_difficulty, "fault": "dropout"},
    }
    for case, perturbations in cases.items():
        fault = perturbations.pop("fault", None)
        for organ in ("liver", "prostate"):
            for controller in ("impedance", "pi", "sta", "blsmc"):
                for seed in seeds:
                    result = run_episode(
                        organ, controller, seed, duration, safety=True,
                        fault=fault, backend=backend, **perturbations,
                    )
                    result["stress_case"] = case
                    rows.append(result)
    _write(output, rows)
    return rows
