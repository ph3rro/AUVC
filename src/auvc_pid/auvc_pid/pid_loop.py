def run_pid(prev_error, current_error, dt, Kp, Ki, Kd, Kf, integral, p_quadratic = False):
    
    #index = len(error) - 1

    #i f len(error) < 0: # if the error list is empty after new target depth is set return feedforward (to hover ideally)
    #    return Kf

    #P = Kp * error[index] ** 2 if p_quadratic else Kp * error[index]
    P = Kp * current_error * abs(current_error) if p_quadratic else Kp * current_error

    integral += current_error * dt

    I = Ki * integral

    if abs(I) > 30.0:
        if (I> 0):
            I = 30.0
        else:
            I = -30.0

    if dt <= 0:
        D = 0
    else:
        D = Kd * ((current_error - prev_error) / dt)


    # if index == 0:
    #     D = 0
    # else:
    #     D = Kd * ((error[index] - error[index - 1]) / dt)

    #print(f"P: {P}, I: {I}, D: {D}, Kf: {Kf}")
    return (integral, P + I + D + Kf)