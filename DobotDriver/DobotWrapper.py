from .DobotDllType import *
import time


class DobotWrapper:
    def __init__(self, COM="COM3"):
        self.COM = COM
        self.simulated = False

    def initiate(self, end_effector_type="cup"):
        self.api = load()

        # connect
        state = ConnectDobot(self.api, self.COM, 115200)[0]
        if not state == DobotConnect.DobotConnect_NoError:
            print("could not connect to DOBOT")
            return False

        # settings
        SetCmdTimeout(self.api, 3000)
        SetQueuedCmdClear(self.api)
        SetQueuedCmdStartExec(self.api)
        device = "DOBOT Magician"
        SetDeviceName(self.api, device)

        SetJOGJointParams(self.api, 50, 50, 50, 50, 50, 50, 50, 50, True)
        SetJOGCoordinateParams(
            self.api, 50, 50, 50, 50, 50, 50, 50, 50, True)
        SetJOGCommonParams(self.api, 100, 100, True)
        SetPTPJointParams(self.api, 100, 100, 100, 100,
                          100, 100, 100, 100, True)
        SetPTPCoordinateParams(self.api, 100, 100, 100, 100, True)
        SetPTPJumpParams(self.api, 20, 100, True)
        SetPTPCommonParams(self.api, 30, 30, True)
        SetHOMEParams(self.api, 200, 0, 0, 0, True)

        if end_effector_type in ["cup", "gripper"]:
            SetEndEffectorParams(self.api, 59.7, 0, 0, 0)
        elif end_effector_type in ["pen"]:
            SetEndEffectorParams(self.api, 61.0, 0, 0, 0)
        else:
            raise ValueError("invalid end effector type")

        # set home
        self.set_home()
        
        return True

    def set_home(self):
        SetHOMECmdEx(self.api, 0, True)

    def reset_alarm(self):
        ClearAllAlarmsState(self.api)

    def move_arm(self, x, y, z, w):
        self._check_xy(x, y)
        if not self.simulated:
            SetPTPCmdEx(self.api, PTPMode.PTPMOVJXYZMode, x, y, z, w, True)

    def get_position(self):
        return GetPose(self.api)[:4]

    def _check_xy(self, x, y):
        r = (x**2+y**2)**0.5

        if r < 115 or r > 320:
            raise ValueError(
                f"invalid position! (x^2+y^2)^0.5 ={r} should be 115~320. x={x} y={y}")

    def cup(self, mode=True):
        if mode:
            SetEndEffectorSuctionCup(self.api, True, True)
        else:
            SetEndEffectorSuctionCup(self.api, False, True)
        # SetEndEffectorGripper(self.api, 0, True)

    def grip(self, grip=True, sleep=0.5):
        if grip:
            SetEndEffectorGripper(self.api, True, True)
        else:
            SetEndEffectorGripper(self.api, True, False)
            time.sleep(sleep)
            SetEndEffectorGripper(self.api, False, False)