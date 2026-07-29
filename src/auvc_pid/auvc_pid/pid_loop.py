def run_pid(error, dt, Kp, Ki, Kd, p_quadratic = False):
    integral = 0

    index = len(error) - 1

    P = Kp * error[index] ** 2 if p_quadratic else Kp * error[index]
    
    for value in error:
        integral += value * dt
    I = Ki * integral


    if index == 0:
        D = 0
    else:
        D = Kd * ((error[index] - error[index - 1]) / dt)


    return P + I + D