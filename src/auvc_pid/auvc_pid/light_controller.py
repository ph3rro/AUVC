from mavros_msgs.msg import OverrideRCIn

NUM_CHANNELS = 18
CHAN_NOCHANGE = 65535

LIGHTS1_INDEX = 8
LIGHTS2_INDEX = 9

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

    def off(self, channel=LIGHTS1_INDEX):
        self._publish(channel, PWM_OFF)

    def full(self, channel=LIGHTS1_INDEX):
        self._publish(channel, PWM_FULL)