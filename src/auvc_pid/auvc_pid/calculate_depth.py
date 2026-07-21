'calculates the depth of the auv depending on the measured pressure of the onboard barometer'
'uses the equation P = pgh, rearranging for h = P/(pg)'

def calculate_depth(measured_pressure):
    output = (measured_pressure - 101325) / (9.81 * 1000)
    return output