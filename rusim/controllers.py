import numpy as np

class Controller:
    def __init__(self, name, target=10.0, gains=None):
        self.name, self.target = name, target
        gains = gains or {}
        self.kp = float(gains.get("kp", .8))
        self.ki = float(gains.get("ki", .15))
        self.k1 = float(gains.get("k1", .8))
        self.k2 = float(gains.get("k2", 2.0))
        self.integral, self.nu = 0., 0.
    def command(self, force, dt, surface_velocity=0., safe=False):
        e = self.target - force
        if self.name == "impedance": u = self.target + self.kp*e - 0.2*surface_velocity
        elif self.name == "pi":
            self.integral = np.clip(self.integral + e*dt, -30, 30)
            u = self.target + self.kp*e + self.ki*self.integral - 0.2*surface_velocity
        elif self.name == "sta":
            s = force-self.target
            self.nu += -self.k2*np.sign(s)*dt
            u = self.target - self.k1*np.sqrt(abs(s)+1e-9)*np.sign(s) + self.nu - .2*surface_velocity
        else: raise ValueError(self.name)
        if safe: u = np.clip(u, 3., 15.)
        return float(np.clip(u, 0., 25.))
