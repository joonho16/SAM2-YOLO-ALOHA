import rospy
from pynput import keyboard
from gello_controller.srv import MoveRobotCartesian, MoveRobotCartesianRequest, MoveRobotCartesianResponse
import moveit_commander
from geometry_msgs.msg import Twist


class TCPcontroller:
    def __init__(self):
        rospy.init_node('tcp_contoller_node')
        rospy.wait_for_service('mover/move_robot_cartesian')
        self.srv = rospy.ServiceProxy('mover/move_robot_cartesian', MoveRobotCartesian)
        self.pub = rospy.Publisher('/mover/cartesian_cmd', Twist, queue_size=10)
    
    def move(self, dir):
        req = MoveRobotCartesianRequest()
        req.x = dir[0]
        req.y = dir[1]
        req.z = dir[2]
        req.roll = dir[3]
        req.pitch = dir[4]
        req.yaw = dir[5]
        response = self.srv(req)

    def move_topic(self, dir):
        twist_msg = Twist()
        twist_msg.linear.x = dir[0]  # x축 선속도 (m/s)
        twist_msg.linear.y = dir[1]  # y축 선속도 (m/s)
        twist_msg.linear.z = dir[2]  # z축 선속도 (m/s)
        twist_msg.angular.x = dir[3]  # x축 각속도 (rad/s)
        twist_msg.angular.y = dir[4]  # y축 각속도 (rad/s)
        twist_msg.angular.z = dir[5]  # z축 각속도 (rad/s)
        self.pub.publish(twist_msg)

key_pressed = set()

def on_press(key):
    try:
        key_pressed.add(key.char)
    except AttributeError:
        print(f"Special key pressed: {key}")

def on_release(key):
    try:
        key_pressed.discard(key.char)
    except AttributeError:
        print(f"Special key released: {key}")

# 키보드 리스너 시작
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

ctr = TCPcontroller()
rate = rospy.Rate(10)  # 10Hz

while not rospy.is_shutdown():
    scale = 0.005
    dir = [0, 0, 0, 0, 0, 0]

    if 'd' in key_pressed:
        dir[0] = +scale
    elif 'a' in key_pressed:
        dir[0] = -scale

    if 'w' in key_pressed:
        dir[1] = +scale
    elif 's' in key_pressed:
        dir[1] = -scale

    if 'e' in key_pressed:
        dir[2] = +scale
    elif 'z' in key_pressed:
        dir[2] = -scale

    if 'l' in key_pressed:
        dir[3] = +scale * 5
    elif '\'' in key_pressed:
        dir[3] = -scale * 5

    if 'p' in key_pressed:
        dir[4] = +scale * 5
    elif ';' in key_pressed:
        dir[4] = -scale * 5

    if '[' in key_pressed:
        dir[5] = +scale * 5
    elif '.' in key_pressed:
        dir[5] = -scale * 5

    if dir != [0, 0, 0, 0, 0, 0]:
        ctr.move(dir)
    # ctr.move_topic(dir)
    # rate.sleep()
# while True:
#     scale = 0.005
#     dir = [0, 0, 0, 0, 0, 0]

#     if 'd' in key_pressed:
#         dir[0] = +scale
#     elif 'a' in key_pressed:
#         dir[0] = -scale

#     if 'w' in key_pressed:
#         dir[1] = +scale
#     elif 's' in key_pressed:
#         dir[1] = -scale

#     if 'e' in key_pressed:
#         dir[2] = +scale
#     elif 'z' in key_pressed:
#         dir[2] = -scale

#     if 'l' in key_pressed:
#         dir[3] = +scale * 5
#     elif '\'' in key_pressed:
#         dir[3] = -scale * 5

#     if 'p' in key_pressed:
#         dir[4] = +scale * 5
#     elif ';' in key_pressed:
#         dir[4] = -scale * 5

#     if '[' in key_pressed:
#         dir[5] = +scale * 5
#     elif '.' in key_pressed:
#         dir[5] = -scale * 5

#     if dir != [0, 0, 0, 0, 0, 0]:
#         ctr.move(dir)

