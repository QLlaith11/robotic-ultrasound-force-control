"""Explicit, unit-tested tissue contact law.

The paper reports force/motion observations, not a released tissue FEM or
constitutive model. This law is therefore a transparent simulation assumption,
not a paper-derived tissue model.
"""
from dataclasses import dataclass
import csv
from pathlib import Path


@dataclass(frozen=True)
class TissueModel:
    stiffness: float = 1000.0
    stiffness_nonlinearity: float = 0.2
    damping: float = 8.0
    force_limit: float = 40.0

    def normal_force(self, penetration: float, surface_velocity: float) -> float:
        if penetration <= 0.0:
            return 0.0
        force = self.stiffness * penetration * (
            1.0 + self.stiffness_nonlinearity * penetration
        ) + self.damping * surface_velocity
        return min(max(0.0, force), self.force_limit)


def validate_tissue_model(output="results/tissue_validation.csv"):
    """Run deterministic constitutive and unit checks before benchmarking."""
    checks = []
    model = TissueModel()
    checks.append(("zero_penetration", model.normal_force(0.0, 0.0) == 0.0))
    checks.append(("negative_penetration", model.normal_force(-0.01, 0.0) == 0.0))
    checks.append(("ten_mm_is_about_ten_newtons",
                   9.9 <= model.normal_force(0.01, 0.0) <= 10.2))
    checks.append(("monotone_penetration",
                   model.normal_force(0.02, 0.0) > model.normal_force(0.01, 0.0)))
    checks.append(("damping_opposes_approach",
                   model.normal_force(0.01, -0.5) < model.normal_force(0.01, 0.0)))
    checks.append(("damping_increases_approach_force",
                   model.normal_force(0.01, 0.5) > model.normal_force(0.01, 0.0)))
    checks.append(("bounded_force", model.normal_force(1.0, 0.0) <= 40.0))
    stiffness_values = [
        TissueModel(stiffness=600.0).normal_force(0.01, 0.0),
        TissueModel(stiffness=1400.0).normal_force(0.01, 0.0),
    ]
    checks.append(("plus_minus_forty_percent_stiffness",
                   5.9 <= stiffness_values[0] <= 6.2
                   and 13.9 <= stiffness_values[1] <= 14.2))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["check", "passed"])
        writer.writerows(checks)
    if not all(passed for _, passed in checks):
        failed = ", ".join(name for name, passed in checks if not passed)
        raise AssertionError(f"tissue validation failed: {failed}")
    return checks
