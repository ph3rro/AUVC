"""
Lights control helper for BlueROV2 (ArduSub) over MAVROS, using RC override
instead of joystick button-function parameters.

Why: BTNn_FUNCTION mappings live on each flight controller and must be
configured per-vehicle. RC channel 9 (Lights1) / channel 10 (Lights2) is the
Blue Robotics factory-default input convention for lights, so it works the
same way across a fleet of stock BlueROV2 units without any per-vehicle setup.

Usage:
    from light_control import LightController

    lights = LightController(node)  # node = your rclpy Node instance

    # anywhere in your existing timer callback:
    lights.set_brightness(75)   # 0-100 %
    lights.off()
    lights.full()
"""

from mavros_msgs.msg import OverrideRCIn

# OverrideRCIn spans 18 channels; entries you don't want to touch must be
# set to CHAN_NOCHANGE, NOT 0 (0 = CHAN_RELEASE, which releases the channel
# back to other input and can cause unexpected behavior on a channel you
# didn't intend to touch).
NUM_CHANNELS = 18
CHAN_NOCHANGE = 65535

# 0-indexed array positions for RC9 / RC10 (i.e. channel 9 -> index 8)
LIGHTS1_INDEX = 8
LIGHTS2_INDEX = 9

# Typical PWM range for BlueROV2 lights. Verify these on your specific
# hardware in QGroundControl's Actuator/Lights test page before relying on
# them -- some units' light drivers may clip slightly differently.
PWM_OFF = 1100
PWM_FULL = 1900


class LightController:
    def __init__(self, node, topic="/override_rc"):
        self._node = node
        self._pub = node.create_publisher(OverrideRCIn, topic, 10)

    def _publish(self, index, pwm_value):
        msg = OverrideRCIn()
        msg.channels = [CHAN_NOCHANGE] * NUM_CHANNELS
        msg.channels[index] = int(pwm_value)
        self._pub.publish(msg)

    def set_brightness(self, percent, channel=LIGHTS1_INDEX):
        """percent: 0-100"""
        percent = max(0, min(100, percent))
        pwm = PWM_OFF + (PWM_FULL - PWM_OFF) * (percent / 100.0)
        self._publish(channel, pwm)

    def off(self, channel=LIGHTS1_INDEX):
        self._publish(channel, PWM_OFF)

    def full(self, channel=LIGHTS1_INDEX):
        self._publish(channel, PWM_FULL)

    # Convenience wrappers if your Heavy config has a second light set on RC10
    def set_brightness_2(self, percent):
        self.set_brightness(percent, channel=LIGHTS2_INDEX)

    def off_2(self):
        self.off(channel=LIGHTS2_INDEX)

    def full_2(self):
        self.full(channel=LIGHTS2_INDEX)