import numpy as np
import pytest
from rusim.kinematics import fk, jacobian
from rusim.patient import PatientMotion
from rusim.simulation import run_episode, bringup, validate_motion, validate_breathing
from rusim.mujoco_backend import MODEL_PATH, make_model
from rusim.tissue import TissueModel, validate_tissue_model
from rusim.experiments import paired_statistics

def test_fk_jacobian_finite_difference():
    q=np.linspace(-.2,.3,7); J=jacobian(q); h=1e-6
    num=np.column_stack([(fk(q+np.eye(7)[i]*h)-fk(q-np.eye(7)[i]*h))/(2*h) for i in range(7)])
    assert np.allclose(J,num,atol=1e-5)

def test_patient_has_event_and_paper_scale():
    m=PatientMotion("liver",3); x=np.array([m.displacement(t) for t in np.linspace(0,60,601)])
    assert x.ptp()*1000 > 5
    assert 3.0 <= m.breathing_period <= 5.0
    assert np.max(np.abs([m.breathing_velocity(t) for t in np.linspace(0,60,601)])) > 0

def test_episode_metrics_and_safety():
    r=run_episode(duration=.2,safety=True)
    assert np.isfinite(r["rmse"]) and r["max_force"] >= 0

def test_mujoco_bringup_and_nominal_estimate():
    assert bringup("results/test_bringup.csv")
    r=run_episode(duration=.02, backend="mujoco", estimator="oracle")
    assert r["backend"] == "mujoco" and np.isfinite(r["estimate_rmse"])

def test_motion_validation_has_forty_seeds():
    rows=validate_motion("results/test_motion.csv")
    assert len(rows) == 80 and {r["organ"] for r in rows} == {"liver","prostate"}

def test_breathing_validation_has_forty_seeds():
    rows=validate_breathing("results/test_breathing_validation.csv")
    assert len(rows) == 80
    assert all(3.0 <= r["period_s"] <= 5.0 and r["peak_to_peak_mm"] > 0 for r in rows)

def test_provenance_robot_and_tissue_models():
    assert MODEL_PATH.is_file()
    model=make_model()
    assert model.nq == model.nv == 7
    tissue=TissueModel()
    assert 9.9 < tissue.normal_force(.01, 0.) < 10.1
    assert tissue.normal_force(-.001, 0.) == 0.

def test_tissue_validation():
    checks = validate_tissue_model("results/test_tissue_validation.csv")
    assert len(checks) == 8 and all(passed for _, passed in checks)

def test_mujoco_backend_is_not_silently_replaced():
    with pytest.raises(ValueError):
        run_episode(duration=.01, backend="not-a-backend")

def test_perturbation_changes_mujoco_result():
    a=run_episode(duration=.05, seed=4, backend="mujoco")
    b=run_episode(duration=.05, seed=4, backend="mujoco", stiffness_scale=.5)
    assert a["backend"] == b["backend"] == "mujoco"
    assert not np.isclose(a["max_force"], b["max_force"])

def test_statistics_pairs_by_organ_and_seed():
    rows=[
        {"organ":"liver","seed":1,"controller":"pi","rmse":1.},
        {"organ":"liver","seed":1,"controller":"sta","rmse":3.},
        {"organ":"prostate","seed":2,"controller":"pi","rmse":10.},
        {"organ":"prostate","seed":2,"controller":"sta","rmse":14.},
    ]
    r=paired_statistics(list(reversed(rows)))
    assert r["n"] == 2 and r["mean_difference"] == 3.

def test_controller_metrics_are_finite():
    r=run_episode(duration=.02, backend="mujoco", estimator="oracle")
    for key in ("mae","error_std","p95_abs_error","overshoot","undershoot","settling_time","retract_rate"):
        assert np.isfinite(r[key])
