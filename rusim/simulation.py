"""MuJoCo-first experiments with an explicit, acceleration-independent tissue model."""
import csv, json
from pathlib import Path
import numpy as np
from .patient import PatientMotion
from .controllers import Controller
from .safety import SafetyMonitor
from .tissue import TissueModel

TARGET = 10.0

def _surrogate_episode(organ, controller, seed, duration, estimator, safety, fault):
    motion=PatientMotion(organ,seed); rng=np.random.default_rng(seed+1000)
    c=Controller(controller); mon=SafetyMonitor(); z=0.; rows=[]; force=0.
    n=int(duration/.002); k=1000*rng.uniform(.6,1.4)
    for i in range(n):
        t=i*.002; surface=motion.displacement(t); sv=motion.velocity(t)
        force=max(0.,k*(surface-z)*(1+.2*max(surface-z,0))+8*sv)+rng.normal(0,.03)
        measured=force+rng.normal(0,.1)
        if fault=="bias" and t>duration*.45: measured+=4
        if fault=="dropout" and duration*.45<t<duration*.45+2: measured=0
        independent=force+rng.normal(0,.16)
        if estimator=="oracle": measured=force
        state=mon.update(measured,independent,.002) if safety else {"estimate":measured,"alarm":False,"retract":False,"cbf":False}
        u=c.command(state["estimate"],.002,sv,safety)
        z += .002*(u+.35*(surface-z)); rows.append((t,force,measured,independent,u,state["alarm"],state["retract"]))
    return _metrics(rows,organ,controller,seed,estimator,safety,"surrogate",fault)

def _metrics(rows,organ,controller,seed,estimator,safety,backend,fault):
    a=np.asarray(rows,float); f=a[:,1]
    error=f-TARGET
    absolute_error=np.abs(error)
    outside=np.flatnonzero(absolute_error > 0.05*TARGET)
    duration=float(a[-1,0]) if len(a) else 0.0
    settling_time=float(a[outside[-1],0]) if len(outside) else 0.0
    return {"organ":organ,"controller":controller,"seed":seed,"estimator":estimator,
            "safety":int(safety),"backend":backend,"fault":fault or "none",
            "rmse":float(np.sqrt(np.mean((f-TARGET)**2))),
            "mae":float(np.mean(absolute_error)),
            "error_std":float(np.std(error)),
            "p95_abs_error":float(np.quantile(absolute_error,.95)),
            "overshoot":float(max(0.,f.max()-TARGET)),
            "undershoot":float(max(0.,TARGET-f.min())),
            "settling_time":settling_time,
            "median":float(np.median(f)),"iqr":float(np.quantile(f,.75)-np.quantile(f,.25)),
            "pct_above_11":float(np.mean(f>11)*100),"pct_above_15":float(np.mean(f>15)*100),
            "max_force":float(f.max()),"effort":float(np.mean(a[:,4]**2)),
            "estimate_rmse":float(np.sqrt(np.mean((a[:,2]-f)**2))),
            "time_above_15":float(np.mean(f>15)*duration),
            "alarm_rate":float(np.mean(a[:,5])),
            "retract_rate":float(np.mean(a[:,6]))}

def _mujoco_episode(organ, controller, seed, duration, estimator, safety, fault,
                    stiffness_scale=1., friction_scale=1., mass_scale=1., torque_noise=1.,
                    gains=None):
    import mujoco
    from .mujoco_backend import make_model
    model=make_model(); model.body_mass[:] *= mass_scale
    model.dof_frictionloss[:] *= friction_scale
    data=mujoco.MjData(model)
    rng=np.random.default_rng(seed+1000); motion=PatientMotion(organ,seed)
    c=Controller(controller, gains=gains); mon=SafetyMonitor(); dt=float(model.opt.timestep)
    # Hold a slightly bent posture while the normal axis is controlled by J^T torque.
    # Avoid the vertical home-pose singularity: the paper's probe is
    # operated in a bent, reachable contact configuration.
    qref=np.array([0.0, 0.5, 0.0, 1.0, 0.0, -1.0, 0.0])
    data.qpos[:]=qref
    data.ctrl[:]=qref
    mujoco.mj_forward(model,data)
    site=model.site("attachment_site").id
    z0=float(data.site_xpos[site,2])
    q_prev=data.qpos.copy(); qvel_prev=data.qvel.copy(); rows=[]
    tissue=TissueModel(stiffness=1000.0*stiffness_scale)
    command_tau=np.zeros(model.nv)
    n=int(duration/dt); control_period=2
    for i in range(n):
        t=i*dt
        # Slow approach: shallow 2 N contact ramped to the 10 N nominal depth.
        base=z0-.010+.008*min(t/1.0,1.0)
        surface=base+motion.displacement(t); sv=motion.velocity(t)
        mujoco.mj_forward(model,data); p=float(data.site_xpos[site,2]); J=np.zeros((3,model.nv))
        mujoco.mj_jacSite(model,data,J,None,site)
        penetration=max(0.,p-surface)
        ftrue=tissue.normal_force(penetration, friction_scale*sv)
        qacc=(data.qvel-qvel_prev)/dt; qvel_prev=data.qvel.copy()
        tissue_tau=-J[2] * ftrue
        tau_sensor=np.zeros(model.nv)
        # Filled after the controller update; the previous command is the
        # measured motor torque for this discrete-time sensor model.
        if estimator=="oracle":
            fhat=ftrue
        else:
            mujoco.mj_forward(model,data)
            mass=np.zeros((model.nv,model.nv)); mujoco.mj_fullM(model,mass,data.qM)
            # Synthesize the motor-torque sensor consistently with the
            # nominal inverse-dynamics equation used by the estimator.
            tau_sensor = (
                mass @ qacc + data.qfrc_bias - tissue_tau
                + rng.normal(0, .04 * torque_noise, model.nv)
            )
            dynamic_rhs=mass @ qacc + data.qfrc_bias
            static_rhs=data.qfrc_bias
            rhs=(dynamic_rhs if estimator=="dynamic" else static_rhs) - tau_sensor
            fhat=float(-np.dot(J[2], rhs) / (np.dot(J[2],J[2])+1e-9))
        independent=ftrue+rng.normal(0,.12)
        if fault=="bias" and t>duration*.45: fhat+=4.
        if fault=="dropout" and duration*.45<t<duration*.45+2: fhat=0.
        state=mon.update(fhat,independent,dt) if safety else {"estimate":fhat,"alarm":False,"retract":False,"cbf":False}
        if i%control_period==0:
            normal=c.command(state["estimate"],dt*control_period,sv,safety)
            if state["retract"]: normal=-.02
            tau=J[2]*normal
            tau += data.qfrc_bias - 8.*(data.qpos-qref)-2.*data.qvel
            command_tau=np.clip(
                tau,
                -np.array([176,176,110,110,110,40,40]),
                np.array([176,176,110,110,110,40,40]),
            )
            data.ctrl[:]=qref
        data.qfrc_applied[:] = tissue_tau + command_tau
        mujoco.mj_step(model,data)
        rows.append((t,ftrue,fhat,independent,float(np.linalg.norm(data.qfrc_applied)),state["alarm"],state["retract"]))
    return _metrics(rows,organ,controller,seed,estimator,safety,"mujoco",fault)

def run_episode(organ="liver",controller="pi",seed=0,duration=60.,estimator="dynamic",
                safety=False,fault=None,backend="mujoco",output=None, **perturbations):
    """Run one episode. MuJoCo errors are intentionally never hidden."""
    if backend == "mujoco":
        result=_mujoco_episode(organ,controller,seed,duration,estimator,safety,fault,**perturbations)
    elif backend == "surrogate":
        result=_surrogate_episode(organ,controller,seed,duration,estimator,safety,fault)
    else:
        raise ValueError("backend must be 'mujoco' or 'surrogate'")
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(result,indent=2))
    return result

def validate_motion(output="results/motion_validation.csv",seeds=range(40)):
    rows=[]
    for organ in ("liver","prostate"):
        for seed in seeds:
            m=PatientMotion(organ,seed); x=np.array([m.displacement(t) for t in np.linspace(0,60,601)])
            rows.append({"organ":organ,"seed":seed,"peak_to_peak_mm":float(np.ptp(x)*1000),
                         "breathing_period_s":float(m.breathing_period),
                         "breathing_amplitude_mm":float(m.breathing_amplitude*1000),
                         "breathing_peak_to_peak_mm":float(np.ptp([m.breathing_displacement(t) for t in np.linspace(0,60,601)])*1000),
                         "breathing_peak_velocity_mm_s":float(np.max(np.abs([m.breathing_velocity(t) for t in np.linspace(0,60,601)]))*1000),
                         "regular_min_mm":5.8,"regular_max_mm":12.1,
                         "paper_max_force_min_N":11.7 if organ=="liver" else 13.4,
                         "paper_max_force_max_N":13.2 if organ=="liver" else 21.3})
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    return rows

def validate_breathing(output="results/breathing_validation.csv",seeds=range(40),duration=60.):
    rows=[]
    times=np.linspace(0.,duration,int(duration*10)+1)
    for organ in ("liver","prostate"):
        for seed in seeds:
            motion=PatientMotion(organ,seed)
            displacement=np.array([motion.breathing_displacement(t) for t in times])
            velocity=np.array([motion.breathing_velocity(t) for t in times])
            rows.append({"organ":organ,"seed":seed,
                         "period_s":float(motion.breathing_period),
                         "frequency_hz":float(motion.breathing_frequency),
                         "amplitude_mm":float(motion.breathing_amplitude*1000),
                         "peak_to_peak_mm":float(np.ptp(displacement)*1000),
                         "peak_velocity_mm_s":float(np.max(np.abs(velocity))*1000),
                         "displacement_rms_mm":float(np.sqrt(np.mean(displacement**2))*1000)})
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    return rows

def bringup(output="results/bringup.csv"):
    import mujoco
    from .mujoco_backend import make_model
    model=make_model(); data=mujoco.MjData(model); data.qpos[:]=np.linspace(-.2,.3,model.nq); mujoco.mj_forward(model,data)
    site=model.site("attachment_site").id; J=np.zeros((3,model.nv)); mujoco.mj_jacSite(model,data,J,None,site)
    approach=_mujoco_episode("liver","pi",7,1., "oracle", False, None)
    checks=[("fk_site_finite",np.isfinite(data.site_xpos[site]).all()),
            ("jacobian_conditioned",np.linalg.cond(J[:2]@J[:2].T)<1e8),
            ("air_force_zero",abs(np.linalg.norm(data.qfrc_applied))<1e-12),
            ("approach_contact_reached",approach["max_force"]>1.0),
            ("stable_contact_hold",np.isfinite(approach["rmse"]) and approach["max_force"]<100.)]
    with Path(output).open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["check","passed"]); w.writerows(checks)
    return all(x[1] for x in checks)

def benchmark(output_dir="results",seeds=range(10),duration=60.,backend="mujoco"):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); validate_motion(out/"motion_validation.csv",range(40)); validate_breathing(out/"breathing_validation.csv",range(40)); bringup(out/"bringup.csv")
    metrics=[]
    for organ in ("liver","prostate"):
        for ctl in ("impedance","pi","sta"):
            for seed in seeds: metrics.append(run_episode(organ,ctl,seed,duration,backend=backend))
    with (out/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=metrics[0]); w.writeheader(); w.writerows(metrics)
    (out/"gains.json").write_text(json.dumps({"tuning_seeds":[100,101,102,103,104],"evaluation_seeds":list(seeds),
        "pi":{"kp":.006,"ki":.0015},"sta":{"k1":.010,"k2":2.0}},indent=2))
    return metrics
