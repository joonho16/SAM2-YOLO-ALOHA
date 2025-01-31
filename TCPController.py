import rospy
from pynput import keyboard
from gello_controller.srv import MoveRobotCartesian, MoveRobotCartesianRequest
from geometry_msgs.msg import Twist
from robotiq_85_msgs.msg import GripperCmd
from sensor_msgs.msg import JointState



class TCPController:
    def __init__(self):
        # rospy.init_node('tcp_contoller_node')
        rospy.wait_for_service('mover/move_robot_cartesian')
        self.srv = rospy.ServiceProxy('mover/move_robot_cartesian', MoveRobotCartesian)
        self.pub = rospy.Publisher('/mover/cartesian_cmd', Twist, queue_size=10)
        self.gripper_pub = rospy.Publisher('gripper/cmd', GripperCmd, queue_size=1)
        rospy.Subscriber('gripper/joint_states', JointState, self.gripper_state_cb)
        self.key_pressed = set()
        self.stop_pressed = False
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.using_keys = ['w', 'a', 's', 'd', 'e', 'z', 'p', 'l', '\'', ';', '[', '.', 'n', 'b']
        self.controlling = False


    def gripper_state_cb(self, msg):
        self.gripper_states = msg


    def move(self, dir):
        req = MoveRobotCartesianRequest()
        req.x = dir[0]
        req.y = dir[1]
        req.z = dir[2]
        req.roll = dir[3]
        req.pitch = dir[4]
        req.yaw = dir[5]
        
        # 서비스 호출 및 응답 저장
        try:
            response = self.srv(req)  # 서비스 호출
            return response           # 서비스 응답 반환
        except Exception as e:
            # 서비스 호출 실패 시 예외 처리
            print(f"Service call failed: {e}")
            return None
        
    def move_gripper(self, value):
        max_gripper_pos = 0.087
        # value = self.gripper_states.position[0] + value

        if max_gripper_pos < value:
            value = max_gripper_pos
        elif value < 0.01:
            value = 0.01

        gripper_pos = GripperCmd()
        gripper_pos.position = value
        self.gripper_pub.publish(gripper_pos)


    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                self.stop_pressed = True
                self.check_controlling()
            if key.char in self.using_keys:
                self.key_pressed.add(key.char)
                self.check_controlling()
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key == keyboard.Key.space:
                self.stop_pressed = False
                self.check_controlling()
            if key.char in self.using_keys:
                self.key_pressed.discard(key.char)
                self.check_controlling()

        except AttributeError:
            pass

    def check_controlling(self):
        if self.stop_pressed or len(self.key_pressed) > 0:
            self.controlling = True
        else:
            self.controlling = False

    def decide_direction(self):
        scale = 0.003
        dir = [0, 0, 0, 0, 0, 0]
        gripper_scale = 0.01
        gripper_dir = 0

        if 'd' in self.key_pressed:
            dir[0] = +scale
        elif 'a' in self.key_pressed:
            dir[0] = -scale

        if 'w' in self.key_pressed:
            dir[1] = +scale
        elif 's' in self.key_pressed:
            dir[1] = -scale

        if 'e' in self.key_pressed:
            dir[2] = +scale
        elif 'z' in self.key_pressed:
            dir[2] = -scale

        if 'l' in self.key_pressed:
            dir[3] = +scale * 5
        elif '\'' in self.key_pressed:
            dir[3] = -scale * 5

        if 'p' in self.key_pressed:
            dir[4] = +scale * 5
        elif ';' in self.key_pressed:
            dir[4] = -scale * 5

        if '[' in self.key_pressed:
            dir[5] = +scale * 5
        elif '.' in self.key_pressed:
            dir[5] = -scale * 5

        if 'n' in self.key_pressed:
            gripper_dir = 0.087
        elif 'b' in self.key_pressed:
            gripper_dir = 0.01

        if dir != [0, 0, 0, 0, 0, 0]:
            self.move(dir)
        if gripper_dir != 0:
            self.move_gripper(gripper_dir)
            

# # 키보드 리스너 시작

# ctr = TCPController()

# while not rospy.is_shutdown():
#     ctr.decide_direction()

