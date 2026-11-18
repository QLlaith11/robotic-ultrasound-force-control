# External model provenance

## Robot

`iiwa7_R800/` is downloaded from
<https://github.com/liuskywalkerjskd/mujoco_iiwa7_R800>, commit `main` at
the time of acquisition. It is a KUKA LBR iiwa 7 R800 MJCF converted from the
ROS `iiwa_stack` URDF and tuned using MuJoCo Menagerie conventions. The source
repository is BSD-3-Clause licensed; its `LICENSE` file is retained here.

The Ipsen et al. study used a KUKA LBR iiwa 7 R800. The repository is therefore
an existing model of the paper's robot, but it is not an official model
released by the paper authors or KUKA. The study must report this distinction.

## Tissue

The paper reports human-subject motion and measured force profiles; it does not
release a tissue FEM, material card, or validated constitutive model. This
study therefore uses `rusim.tissue.TissueModel`, an explicit Kelvin-Voigt-like
normal penalty law with stiffness in N/m, documented parameters, and a 40 N
numerical envelope. It is a simulation assumption calibrated only to the
reported force/motion ranges, not an existing tissue model from the paper.
