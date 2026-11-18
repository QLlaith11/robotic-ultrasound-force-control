import numpy as np

class SafetyMonitor:
    def __init__(self, lower=3., upper=15., hard=20., persistence=0.02):
        self.lower, self.upper, self.hard, self.persistence = lower, upper, hard, persistence
        self.bad_time, self.fault_time, self.fallback = 0., 0., False
    def update(self, estimate, independent, dt):
        inconsistent = abs(estimate-independent) > 2.0
        self.bad_time = self.bad_time + dt if inconsistent else 0.
        if self.bad_time >= self.persistence: self.fallback = True
        return {"estimate": independent if self.fallback else estimate,
                "retract": estimate > self.hard, "alarm": self.fallback,
                "cbf": estimate < self.lower or estimate > self.upper}
