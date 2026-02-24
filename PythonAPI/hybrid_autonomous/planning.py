# Planning and control module (classical)
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        P = self.Kp * error
        self.integral += error * self.dt
        I = self.Ki * self.integral
        D = self.Kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        return P + I + D

class Planner:
    def __init__(self):
        self.controller = PIDController(1.0, 0.0, 0.1, 0.05)

    def compute_control(self, target, current):
        error = target - current
        return self.controller.update(error)
