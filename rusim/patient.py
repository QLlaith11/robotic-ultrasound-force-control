import numpy as np

PAPER_RANGES_MM = {
    "liver": {"regular": (5.8, 10.1), "drift": (3.2, 9.9), "max_force": (11.7, 13.2)},
    "prostate": {"regular": (4.6, 10.1), "drift": (3.2, 9.9), "max_force": (13.4, 21.3)},
}

class PatientMotion:
    """Paper-calibrated organ-surface motion with an explicit respiratory component."""
    def __init__(self, organ="liver", seed=0, event_compression=20.0):
        if organ not in PAPER_RANGES_MM: raise ValueError(organ)
        self.organ, self.rng = organ, np.random.default_rng(seed)
        self.event_compression = event_compression
        self.breathing_period = self.rng.uniform(3.0, 5.0)
        self.breathing_amplitude = self.rng.uniform(*PAPER_RANGES_MM[organ]["regular"]) * 1e-3
        self.drift = self.rng.uniform(*PAPER_RANGES_MM[organ]["drift"]) * 1e-3
        self.shift_time = self.rng.uniform(8.0, 35.0)
        self.shift = self.rng.choice([-1, 1]) * self.breathing_amplitude

    @property
    def breathing_frequency(self):
        return 1.0 / self.breathing_period

    def breathing_displacement(self, t):
        return self.breathing_amplitude * (
            0.65*np.sin(2*np.pi*t/self.breathing_period) +
            0.25*np.sin(4*np.pi*t/self.breathing_period + .4) +
            0.10*np.sin(6*np.pi*t/self.breathing_period)
        )

    def displacement(self, t):
        b = self.breathing_displacement(t)
        cardiac = 0.00025*np.sin(2*np.pi*1.1*t)
        drift = self.drift * min(t/60.0, 1.0) * 0.35
        shift = self.shift * (1.0 - np.exp(-max(t-self.shift_time, 0.0)/.08))
        cough = (0.004*np.sin(2*np.pi*4*t) * np.exp(-((t-(self.shift_time+2))/0.25)**2)
                 if self.organ == "prostate" else 0.0)
        return b + cardiac + drift + shift + cough

    def breathing_velocity(self, t):
        h = 1e-4
        return (self.breathing_displacement(t+h)-self.breathing_displacement(t-h))/(2*h)

    def velocity(self, t):
        h = 1e-4
        return (self.displacement(t+h)-self.displacement(t-h))/(2*h)
