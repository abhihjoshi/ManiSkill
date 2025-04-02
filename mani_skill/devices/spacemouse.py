"""
Spacemouse controller currently for the floating panda gripper.
"""
import threading
import math
import time

import hid
import numpy as np
from pynput.keyboard import Controller, Key, Listener

def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.asarray(data)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    E.g.:
        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True

        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    Args:
        angle (float): Magnitude of rotation
        direction (np.array): (ax,ay,az) axis about which to rotate
        point (None or np.array): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        np.array: 4x4 homogeneous matrix that includes the desired rotation
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.asarray(point[:3], dtype=np.float32)
        M[:3, 3] = point - np.dot(R, point)
    return M

def to_int16(y1, y2):
    """
    Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1 (int): 8-bit byte
        y2 (int): 8-bit byte

    Returns:
        int: 16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x

def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """
    Normalize raw HID readings to target range.

    Args:
        x (int): Raw reading from HID
        axis_scale (float): (Inverted) scaling factor for mapping raw input value
        min_v (float): Minimum limit after scaling
        max_v (float): Maximum limit after scaling

    Returns:
        float: Clipped, scaled input from HID
    """
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """
    Converts SpaceMouse message to commands.

    Args:
        b1 (int): 8-bit byte
        b2 (int): 8-bit byte

    Returns:
        float: Scaled value from Spacemouse message
    """
    return scale_to_control(to_int16(b1, b2))

class SpaceMouse:

    def __init__(self,
                 vendor_id=9583,
                 product_id=50741,
                 pos_sensitivity=1.0,
                 rot_sensitivity=1.0):
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = hid.device()
        try:
            self.device.open(self.vendor_id, self.product_id)  # SpaceMouse
        except OSError as e:
            raise

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        # also add a keyboard for aux controls
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()
    
    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "reset simulation")
        print_command("Left button (hold)", "close gripper")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command("Twist mouse about an axis", "rotate arm about a corresponding axis")
        print_command("Control+C", "quit")
        print_command("b", "toggle arm/base mode (if applicable)")
        print_command("s", "switch active arm (if multi-armed robot)")
        print_command("=", "switch active robot (if multi-robot environment)")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(6)
        # Reset grasp
        self.single_click_and_hold = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the 3D mouse.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
        )
    
    def run(self):
        """Listener method that keeps pulling new messages."""

        t_last_click = -1

        while True:
            d = self.device.read(13)
            if d is not None and self._enabled:

                ## logic for older spacemouse model

                if d[0] == 1:  ## readings from 6-DoF sensor
                    self.y = convert(d[1], d[2])
                    self.x = convert(d[3], d[4])
                    self.z = convert(d[5], d[6]) * -1.0

                elif d[0] == 2:

                    self.roll = convert(d[1], d[2])
                    self.pitch = convert(d[3], d[4])
                    self.yaw = convert(d[5], d[6])

                    self._control = [
                        self.x,
                        self.y,
                        self.z,
                        self.roll,
                        self.pitch,
                        self.yaw,
                    ]

                if d[0] == 3:  ## readings from the side buttons

                    # press left button
                    if d[1] == 1:
                        t_click = time.time()
                        elapsed_time = t_click - t_last_click
                        t_last_click = t_click
                        self.single_click_and_hold = True

                    # release left button
                    if d[1] == 0:
                        self.single_click_and_hold = False

                    # right button is for reset
                    if d[1] == 2:
                        self._reset_state = 1
                        self._enabled = False
                        self._reset_internal_state()

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.

        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """
        pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        pass

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 50
        dpos = dpos * 125

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation
    
    def input2action(self, mirror_actions=False):
        state = self.get_controller_state()
        dpos, rotation, raw_drotation, grasp, reset = (
            state["dpos"],
            state["rotation"],
            state["raw_drotation"],
            state["grasp"],
            state["reset"],
        )
        if reset:
            return None
        
        drotation = raw_drotation[[1, 0, 2]]
        drotation[2] = -drotation[2]
        dpos, drotation = self._postprocess_device_outputs(dpos, drotation)
        grasp = -1 if grasp else 1

        arm_norm_delta = np.concatenate([dpos, drotation])

        ac_dict = {
            "delta_pos": dpos,
            "delta_rot": drotation
        }
        # arm_norm_delta = np.concatenate([dpos, drotation])
        
        # arm_action = self.get_arm_action(
        #     robot,
        #     active_arm,
        #     norm_delta=arm_norm_delta,
        # )
        
        # ac_dict[f"{active_arm}_abs"] = arm_action["abs"]
        # ac_dict[f"{active_arm}_delta"] = arm_action["delta"]

        # for (k, v) in ac_dict.items():
        #     if "abs" not in k:
        #         ac_dict[k] = np.clip(v, -1, 1)
        
        return ac_dict
        
    # def get_arm_action(self):
    #     arm_controller = robot.part_controllers[arm]
    #     delta_action = arm_controller.scale_action(norm_delta.copy())
    #     abs_action = arm_controller.delta_to_abs_action(delta_action, goal_update_mode=None)
    #     return {
    #         "delta": norm_delta,
    #         "abs": abs_action,
    #     }
        

