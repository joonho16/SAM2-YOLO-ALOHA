import rospy
from pynput import keyboard
from gello_controller.srv import MoveRobotCartesian, MoveRobotCartesianRequest
from geometry_msgs.msg import Twist
from robotiq_85_msgs.msg import GripperCmd
from sensor_msgs.msg import JointState
from utils import rescale_val



class TCPController:
    def __init__(self, gripper_pub, get_gripper_state_fn):
        rospy.wait_for_service('mover/move_robot_cartesian')
        self.srv = rospy.ServiceProxy('mover/move_robot_cartesian', MoveRobotCartesian)
        self.pub = rospy.Publisher('/mover/cartesian_cmd', Twist, queue_size=10)
        self.get_gripper_state = get_gripper_state_fn
        self.gripper_pub = gripper_pub
        self.key_pressed = set()
        self.stop_pressed = False
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.using_keys = ['w', 'a', 's', 'd', 'e', 'z', 'p', 'l', '\'', ';', '[', '.', 'n', 'b']
        self.controlling = False


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
        # 그리퍼 캘리브레이션
        gripper_state = self.get_gripper_state()
        
        target_gripper_pos = gripper_state.position[0] + value

        if 0.087 < target_gripper_pos:
            target_gripper_pos = 0.087
        elif target_gripper_pos < 0:
            target_gripper_pos = 0

        gripper_pos_cmd = GripperCmd()
        gripper_pos_cmd.position = target_gripper_pos

        self.gripper_pub.publish(gripper_pos_cmd)


    def on_press(self, key):
        try:
            # if key == keyboard.Key.space:
            #     self.stop_pressed = True
            #     self.check_controlling()
            if key.char in self.using_keys:
                self.key_pressed.add(key.char)
                self.check_controlling()
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            # if key == keyboard.Key.space:
            #     self.stop_pressed = False
            #     self.check_controlling()
            if key.char in self.using_keys:
                self.key_pressed.discard(key.char)
                # self.check_controlling()

        except AttributeError:
            pass

    def check_controlling(self):
        if self.stop_pressed or len(self.key_pressed) > 0:
            self.controlling = True
        else:
            self.controlling = False

    def decide_direction(self):
        scale = 0.01
        dir = [0, 0, 0, 0, 0, 0]
        gripper_scale = 0.01
        gripper_dir = 0
        
        key_check = False

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
            gripper_dir = + gripper_scale
        elif 'b' in self.key_pressed:
            gripper_dir = - gripper_scale

        if dir != [0, 0, 0, 0, 0, 0]:
            self.move(dir)
            key_check = True
        if gripper_dir != 0:
            self.move_gripper(gripper_dir)
            key_check = True
        
        return key_check
            

# # 키보드 리스너 시작

if __name__ == '__main__':
    rospy.init_node('tcp_contoller_node')
    ctr = TCPController()

    while not rospy.is_shutdown():
        ctr.decide_direction()

