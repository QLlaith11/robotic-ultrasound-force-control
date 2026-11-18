# Summary

The package now defaults to a provenance-tracked MuJoCo KUKA LBR iiwa 7 R800
model converted from the ROS `iiwa_stack` URDF, with
1 ms integration and 2 ms control updates. Tissue is an explicit penalty
force injected through `qfrc_applied`; sensing reconstructs force from noisy
joint torques, with static/dynamic/oracle variants and an independent monitor.
It includes paired liver/prostate benchmarks, held-out tuning metadata,
40-seed motion and respiratory validation, bring-up checks,
PI/STA/safety/fault/robustness frameworks, CSV output, and paired
bootstrap/Wilcoxon statistics. Controller comparisons report RMSE, mean
absolute error, error variability, 95th-percentile error, overshoot,
undershoot, settling time, threshold exposure, effort, alarms, retractions,
and estimator error.

The patient-motion model exposes a randomized 3–5 s respiratory period and
organ-specific breathing amplitude, velocity, drift, cardiac ripple, and
abrupt events. The paper does not release a tissue FEM or constitutive model,
law is explicitly labeled as a simulation assumption rather than an existing
paper asset.

Validated here: 10 tests pass; the imported model and tissue law validation
pass; and the full 10-seed, 60 s-per-episode MuJoCo suite completed,
producing 60 paired evaluation rows plus estimation, safety/fault, robustness,
motion-validation, and statistical result tables. The corrected benchmark
holds the 10 N target closely: mean RMSE is 0.238/0.336/0.261 N for
impedance/PI/STA in liver and 0.587/0.462/0.540 N in prostate. Results are
not clinical evidence: motion events are compressed x20, the imported arm
asset is not an official paper/KUKA release, and ROS 2,
hardware-in-the-loop, calibration, and in-vivo validation remain.
