from auvc_pid.pid_loop import *

def calculate_heave(errors, dt):
    Kp = 3.6
    Ki = 0.0
    Kd = 1.2

    multiplier = 25.0
    pid_raw = run_pid(errors, dt, Kp, Ki, Kd)

    heave = pid_raw * multiplier

    if (abs(heave) > 400.0):
        if (heave > 0):
            return -400.0
        else:
            return 400.0

    return -heave

    