"""MuJoCo backend using a provenance-tracked KUKA LBR iiwa 7 R800 model."""
from pathlib import Path
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "iiwa7_R800" / "iiwa7_mjcf" / "iiwa7_tuned.xml"
SITE_NAME = "attachment_site"


def make_model():
    """Load the downloaded iiwa 7 R800 MJCF and enforce the study timestep."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"iiwa 7 R800 model is missing: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = 0.001
    # The imported MJCF exposes stiff position actuators. The study evaluates
    # torque control from joint measurements, so disable those actuator
    # torques while retaining the imported masses, inertias, limits, and
    # geometry; controller torques are applied explicitly via qfrc_applied.
    model.actuator_gainprm[:, 0] = 0.0
    model.actuator_biasprm[:, 1:3] = 0.0
    if model.nq != 7 or model.nv != 7:
        raise ValueError(f"expected a 7-DoF iiwa model, got nq={model.nq}, nv={model.nv}")
    if model.site(SITE_NAME).id < 0:
        raise ValueError(f"model has no {SITE_NAME!r} site")
    return model
