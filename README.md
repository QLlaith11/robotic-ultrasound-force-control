# Robotic ultrasound force-control study

This package is a reproducible MuJoCo-first study. It uses a 7-DoF
KUKA LBR iiwa 7 R800 MJCF model downloaded from the BSD-3-Clause
`mujoco_iiwa7_R800` repository and converted from the ROS `iiwa_stack` URDF,
1 ms MuJoCo integration, 2 ms controller updates, and explicit compliant
tissue force injected through `qfrc_applied`.
The nominal-vs-true force estimate includes noisy torque sensing and an
independent consistency estimate. The patient generator is calibrated to the
supplied paper context: 10 N target,
liver maxima 11.7–13.2 N, prostate maxima 13.4–21.3 N, and 5.8–12.1 mm
regular motion. Abrupt events are compressed by x20 (explicitly not clinical
time).

Commands:

* `python -m rusim.cli --fast --output results_fast` (6 s, 10 paired seeds)
* `python -m rusim.cli --rigorous --output results` (60 s, 10 paired seeds,
  held-out tuning, motion validation, estimation/safety/robustness studies)
* `python -m rusim.cli --rigorous --seeds 2 --duration 1 --output results_small`
  is a bounded end-to-end smoke suite.

`results/` contains CSVs, saved gains, paired Wilcoxon/bootstrap statistics,
bring-up checks, motion validation, and SVG figures. `--backend surrogate`
selects the clearly labeled NumPy fallback.

The private LüGloMMa application package is kept outside this public
repository at `/home/laith/Downloads/lueglomma_application_private/`. It
contains the research proposal, motivation-letter draft, two-page CV
structure, technical brief, and submission checklist. These documents target
the Lübeck Innovation Hub for Robotic Surgery and the MSc Robotics and
Autonomous Systems. Replace all `[[...]]` placeholders with verifiable
applicant information before use.

## Controller metrics

`results/metrics.csv` reports one row per organ, controller, and seed.
`rmse`, `mae`, `error_std`, and `p95_abs_error` quantify tracking accuracy and
tail risk. `overshoot`, `undershoot`, and `settling_time` quantify transient
behavior. `pct_above_11`, `pct_above_15`, and `time_above_15` quantify force
exposure. `effort` measures mean squared applied torque norm, while
`alarm_rate` and `retract_rate` quantify safety-monitor intervention.
`paired_statistics.csv` compares PI and super-twisting results by matched
organ/seed with bootstrap intervals and a Wilcoxon test.

`results/breathing_validation.csv` reports respiratory period, frequency,
amplitude, peak-to-peak displacement, peak velocity, and displacement RMS for
80 deterministic organ/seed cases. The breathing component is a
non-sinusoidal multi-harmonic surface motion with a randomized 3–5 s period.

## Visual outputs

The public results include a controller dashboard
(`results/figures/controller_dashboard.svg`), respiratory-motion plot
(`results/figures/respiratory_motion.svg`), and individual metric figures.
The animation in `results/animation/robot_breathing_contact.mp4` and
`robot_breathing_contact.gif` is rendered by MuJoCo's offscreen renderer from
the imported iiwa MJCF. It uses the same controller-to-force-to-joint-torque
path as the benchmark, and the companion
`robot_breathing_force_control.csv` records true force, estimated force, and
commanded normal force. It is a dynamic simulation visualization, not a
recorded hardware experiment.

Regenerate them with:

```bash
python3 scripts/make_public_plots.py
python3 scripts/render_animation.py --output-dir results/animation
```

Limitations: this remains simulation, the imported arm model is not an
official asset released by the paper authors or KUKA, force sensing is
reconstructed from simulated joint torques, and event compression changes
temporal statistics. The paper does not release a tissue FEM or constitutive
model; the explicit tissue law is a transparent simulation assumption. The
robustness sweep now applies stiffness, friction, mass, and torque-noise
perturbations. The explicit penalty contact includes a documented 40 N
numerical/physical envelope to prevent unstable rigid-body excursions from
becoming kilonewton artifacts; this is not a measured tissue limit. It is
still a compact model rather than hardware evidence.
Results do not
replace ROS 2, hardware-in-the-loop, calibration, or clinical validation.
