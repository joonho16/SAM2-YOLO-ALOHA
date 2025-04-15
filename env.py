import rospy
import threading
import math
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
import time
import dm_env
import collections
from cv_bridge import CvBridge
from constants import DT, DEFAULT_CAMERA_NAMES, JOINT_LIMIT, TOPIC_NAME, JOINT_NAMES, TOOL_NAMES
from open_manipulator_msgs.srv import SetJointPositionRequest, SetJointPosition
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from robotiq_85_msgs.msg import GripperCmd
from gello_controller.srv import MoveRobot, MoveRobotRequest
import actionlib
from PIL import Image as PILImage
import numpy as np
import sys
import kinematics
from utils import qpos_to_xpos, xpos_to_qpos, ros_image_to_numpy, rescale_val


class AlohaEnv:
    def __init__(self, kin_config, camera_names=DEFAULT_CAMERA_NAMES, robot_name="ur5"):
        self.robot_name = robot_name
        self.joint_states = None
        self.camera_names = camera_names
        # self.cam_1_image = None
        # self.cam_2_image = None
        # self.cam_3_image = None
        # self.digit_image = None
        self.js_mutex = threading.Lock()
        self.bridge = CvBridge()
        self.is_showing = False
        self.joint_names = JOINT_NAMES[robot_name]
        self.pyversion = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if robot_name == 'ur5':
            rospy.Subscriber('ur5e/joint_states', JointState, self.joint_state_cb)
            rospy.Subscriber('ur5e/ur5e_scaled_pos_joint_traj_controller/command', JointTrajectory, self.master_joint_state_cb)
            rospy.Subscriber('gripper/joint_states', JointState, self.gripper_state_cb)
            rospy.Subscriber('gripper/cmd', GripperCmd, self.master_gripper_state_cb)
            self.move_robot = actionlib.SimpleActionClient('/ur5e/ur5e_scaled_pos_joint_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
            self.move_robot.wait_for_server()
            self.gripper_publisher = rospy.Publisher('gripper/cmd', GripperCmd, queue_size=1)

        elif robot_name == 'om':
            rospy.Subscriber("/joint_states", JointState, self.joint_state_cb)
            rospy.Subscriber("/master_joint_states", JointState, self.master_joint_state_cb)

        for cam_name in camera_names:
            if 'digit' in cam_name:
                rospy.Subscriber(f"/{cam_name}/image_raw", Image, self.digit_image_raw_cb, callback_args=cam_name)
            else:
                rospy.Subscriber(f"/{cam_name}/color/image_raw/compressed", CompressedImage, self.image_raw_cb, callback_args=cam_name)
                
        self.kn = kinematics.Kinematics(kin_config)

        time.sleep(0.1)
        # rospy.spin()


    def joint_state_cb(self, msg):
        with self.js_mutex:
            self.joint_states = msg
    
    def gripper_state_cb(self, msg):
        with self.js_mutex:
            if self.robot_name == 'ur5':
                self.gripper_states = msg
                self.gripper_states.position = (rescale_val(msg.position[0], (0.0, 0.786), (0.087, 0.001)),)
            else:
                self.gripper_states = msg

    def master_joint_state_cb(self, msg):
        with self.js_mutex:
            self.master_joint_states = msg

    def master_gripper_state_cb(self, msg):
        with self.js_mutex:
            self.master_gripper_states = msg
    
    def image_raw_cb(self, data, cam_name):
        image = ros_image_to_numpy(data)

        setattr(self, cam_name, image)

    def digit_image_raw_cb(self, data, cam_name):
        cv2img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        pilimg = PILImage.fromarray(cv2img)

        resized_img = np.array(pilimg.resize((640, 480), PILImage.BILINEAR))

        setattr(self, cam_name, resized_img)


    # def image_raw_cb_4(self, data):
    #     cv2img = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    #     pilimg = PILImage.fromarray(cv2img)


    #     resized_img = pilimg.resize((1280, 720), PILImage.BILINEAR)

    #     filtered_img = np.load('tactile_default.npy') - resized_img

    #     setattr(self, 'digit_image', filtered_img)

    def get_reward(self):
        return 0
    
    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['xpos'] = self.get_xpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs
    
    def reset(self):
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())
    
    def record_step(self):
        time.sleep(DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())
    
    def move_step_om(self, action):        
        rospy.wait_for_service('goal_joint_space_path')
        rospy.wait_for_service('goal_tool_control')
        try:
            goal_joint_space_path = rospy.ServiceProxy("goal_joint_space_path", SetJointPosition)
            goal_tool_control = rospy.ServiceProxy("goal_tool_control", SetJointPosition)
            srv_request1 = SetJointPositionRequest()
            srv_request2 = SetJointPositionRequest()

            state_len = len(JOINT_NAMES[self.robot_name])
            srv_request1.joint_position.joint_name = self.joint_names

            for i, joint_name in enumerate(self.joint_names):
                if action[i] > JOINT_LIMIT[self.robot_name][joint_name]['max']:
                    print(f"Limit over at {joint_name}: {action[i]}")
                    action[i] = JOINT_LIMIT[self.robot_name][joint_name]['max']
                elif action[i] < JOINT_LIMIT[self.robot_name][joint_name]['min']:
                    print(f"Limit over at {joint_name}: {action[i]}")
                    action[i] = JOINT_LIMIT[self.robot_name][joint_name]['min']

            srv_request1.joint_position.position = action[:state_len]
            srv_request1.path_time = 1.5


            srv_request2.joint_position.joint_name = ['gripper']
            if action[state_len] > JOINT_LIMIT[self.robot_name]['gripper']['max']:
                gripper_state = JOINT_LIMIT[self.robot_name]['gripper']['max']
            elif action[state_len] < JOINT_LIMIT[self.robot_name]['gripper']['min']:
                gripper_state = JOINT_LIMIT[self.robot_name]['gripper']['min']
            else:
                gripper_state = action[state_len]
            srv_request2.joint_position.position = [gripper_state]

            srv_request2.path_time = 1.5

            response1 = goal_joint_space_path(srv_request1)
            # print('joint')
            response2 = goal_tool_control(srv_request2)
            # print('tool')
            if response1.is_planned and response2.is_planned:
                return dm_env.TimeStep(
                    step_type=dm_env.StepType.MID,
                    reward=self.get_reward(),
                    discount=None,
                    observation=self.get_observation())
            else:
                print('Service call failed')
                exit()
            
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            return False
        
    def move_step_ur5(self, action):

        try:
            joint_trajectory = JointTrajectory()
            joint_trajectory.joint_names = ["ur5e_elbow_joint", "ur5e_shoulder_lift_joint", "ur5e_shoulder_pan_joint", "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint"]

            rad_pos = action[:6]

            # rospy.loginfo(f"[0] {rad_pos[0]} [1] {rad_pos[1]} [2] {rad_pos[2]} [3] {rad_pos[3]} [4] {rad_pos[4]} [5] {rad_pos[5]}")

            joint_point = JointTrajectoryPoint()
            joint_point.positions = rad_pos

            joint_point.time_from_start = rospy.Duration(0.5)

            joint_trajectory.points.append(joint_point)

            joint_goal = FollowJointTrajectoryGoal()
            joint_goal.trajectory = joint_trajectory

            self.move_robot.send_goal(joint_goal)

            #-------------------------------------Gripper

            gripper_pos = GripperCmd()
            gripper_pos.position = action[-1]
            self.gripper_publisher.publish(gripper_pos)
            # self.move_robot.wait_for_result()
            # result = self.move_robot.get_result()

            # if result.error_code == 0:
            # rospy.loginfo("Success: Trajectory executed successfully.")
            return dm_env.TimeStep(
                step_type=dm_env.StepType.MID,
                reward=self.get_reward(),
                discount=None,
                observation=self.get_observation()
            )
            # else:
            #     rospy.logwarn(f"Failed: {result.error_string}")
            #     exit()

        except rospy.ROSException as e:
            rospy.logerr("Action call failed: %s", e)
            return False

        
    def move_step(self, action):
        if self.robot_name == 'om':
            return self.move_step_om(action)
        elif self.robot_name == 'ur5':
            return self.move_step_ur5(action)

    def get_qpos(self):
        if self.robot_name == 'om':
            return self.joint_states.position
        elif self.robot_name == 'ur5':
            return self.joint_states.position + self.gripper_states.position
        
    def get_xpos(self):
        if self.robot_name == 'om':
            return self.joint_states.position
        elif self.robot_name == 'ur5':
            # qpos = self.joint_states.position
            # qpos[0], qpos[2] = qpos[2], qpos[0]
            # xpos_only = self.kn.forward_kinematics(qpos)
            # xpos_only = xpos_only.cpu().detach().numpy()
            # ee_pos = np.array(self.gripper_states.position)
            # xpos = np.concatenate((xpos_only, ee_pos), axis=0)
            xpos = qpos_to_xpos(self.get_qpos(), self.kn)
            return xpos
        
    def get_xvel(self, last_xpos):
        if self.robot_name == 'om':
            return self.joint_states.position
        elif self.robot_name == 'ur5':
            xpos = qpos_to_xpos(self.get_qpos(), self.kn)
            xvel = xpos - last_xpos
            return xvel

    def get_qvel(self):
        if self.robot_name == 'om':
            return self.joint_states.velocity
        elif self.robot_name == 'ur5':
            return self.joint_states.velocity + self.gripper_states.velocity
    
    def get_effort(self):
        if self.robot_name == 'om':
            return self.joint_states.effort
        elif self.robot_name == 'ur5':
            return self.joint_states.effort + (0,)
    
    def get_action(self):
        if self.robot_name == 'om':
            return self.master_joint_states.position
        elif self.robot_name == 'ur5':
            return self.master_joint_states.points[0].positions + (self.master_gripper_states.position,)
        
    def get_xaction(self):
        if self.robot_name == 'om':
            return self.master_joint_states.position
        elif self.robot_name == 'ur5':
            # xaction_only = self.kn.forward_kinematics(self.master_joint_states.points[0].positions).cpu().numpy()
            # ee_action = np.array(self.master_gripper_states.position)
            # xaction = np.concatenate((xaction_only, ee_action), axis=0)  # axis=0으로 설정
            xaction = qpos_to_xpos(self.get_action(), self.kn)
            return xaction
        
    def get_xvel_action(self, last_xaction):
        if self.robot_name == 'om':
            return self.master_joint_states.position
        elif self.robot_name == 'ur5':
            xaction = qpos_to_xpos(self.get_action(), self.kn)
            xvel_action = xaction - last_xaction
            return xvel_action

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}')
        return image_dict
    
    def go_home_pose_om(self, is_training=True):
        if(is_training):
            rospy.wait_for_service('goal_master_joint_space_path')
        else:
            rospy.wait_for_service('goal_joint_space_path')
            rospy.wait_for_service('goal_tool_control')
        try:

            if(is_training):
                srv_request = SetJointPositionRequest()
                goal_joint_space_path = rospy.ServiceProxy("goal_master_joint_space_path", SetJointPosition)
                srv_request.path_time = 2.0
                srv_request.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']
                srv_request.joint_position.position = [0.0, -1.05, 0.35, 0.70, -0.01]
                response = goal_joint_space_path(srv_request)
                return response.is_planned  
            else: 
                srv_request1 = SetJointPositionRequest()
                srv_request2 = SetJointPositionRequest()
                goal_joint_space_path = rospy.ServiceProxy("goal_joint_space_path", SetJointPosition)
                goal_tool_control = rospy.ServiceProxy("goal_tool_control", SetJointPosition)
                srv_request1.path_time = 2.0
                srv_request1.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4']
                srv_request1.joint_position.position = [0.0, -1.05, 0.35, 0.70]

                srv_request2.joint_position.joint_name = ['gripper']
                srv_request2.joint_position.position = [-0.01]
                srv_request2.path_time = 1.5
            
                response1 = goal_joint_space_path(srv_request1)
                response2 = goal_tool_control(srv_request2)

                if response1.is_planned and response2.is_planned:
                    time.sleep(3)
                    return dm_env.TimeStep(
                        step_type=dm_env.StepType.MID,
                        reward=self.get_reward(),
                        discount=None,
                        observation=self.get_observation())
                else:
                    return False
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
            return False

    def go_home_pose_ur5(self, is_training=True):
        if is_training:
            time.sleep(8)
            return True
        else:
            goal_gripper_pos = 0.087
            gripper_pos = GripperCmd()
            current_gripper_pos = self.gripper_states.position[0]
            gradient = (goal_gripper_pos - current_gripper_pos) / 20
            while current_gripper_pos < goal_gripper_pos:
                current_gripper_pos += gradient
                gripper_pos.position = current_gripper_pos
                self.gripper_publisher.publish(gripper_pos)
            rospy.wait_for_service('mover/move_robot_planning')
            try:
                move_robot = rospy.ServiceProxy('mover/move_robot_planning', MoveRobot)


                joint_trajectory = JointTrajectory()
                joint_trajectory.joint_names = ["ur5e_elbow_joint", "ur5e_shoulder_lift_joint", "ur5e_shoulder_pan_joint", "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint"]

                # rad_pos = [-1.62, -1.45, -1.45, -1.23, 1.32, -1.52]
                rad_pos = [-2.29, -1.95, -1.66, -0.46, 1.71, -1.62]


                rospy.loginfo(f"[0] {rad_pos[0]} [1] {rad_pos[1]} [2] {rad_pos[2]} [3] {rad_pos[3]} [4] {rad_pos[4]} [5] {rad_pos[5]}")

                request = MoveRobotRequest()
                request.joint_trajectory = rad_pos[:6]
                # print(joint_trajectory)

                response = move_robot(request)

                if response.success:
                    time.sleep(3)
                    return dm_env.TimeStep(
                        step_type=dm_env.StepType.MID,
                        reward=self.get_reward(),
                        discount=None,
                        observation=self.get_observation())
                else:
                    rospy.logwarn("Failed: %s", response.message)
                    exit()

            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s", e)
                return False

    def go_home_pose(self, is_training=True):
        if self.robot_name == 'om':
            return self.go_home_pose_om(is_training)
        elif self.robot_name == 'ur5':
            return self.go_home_pose_ur5(is_training)
        
    
    def move_joint(self, rad_pos):
        goal_gripper_pos = rad_pos[6]
        goal_joint_pos = rad_pos[:6]
        gripper_pos = GripperCmd()
        gripper_pos.position = goal_gripper_pos
        self.gripper_publisher.publish(gripper_pos)
        rospy.wait_for_service('mover/move_robot_planning')
        move_robot = rospy.ServiceProxy('mover/move_robot_planning', MoveRobot)

        joint_trajectory = JointTrajectory()
        joint_trajectory.joint_names = ["ur5e_elbow_joint", "ur5e_shoulder_lift_joint", "ur5e_shoulder_pan_joint", "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint"]

        rospy.loginfo(f"[0] {goal_joint_pos[0]} [1] {goal_joint_pos[1]} [2] {goal_joint_pos[2]} [3] {goal_joint_pos[3]} [4] {goal_joint_pos[4]} [5] {goal_joint_pos[5]}")

        request = MoveRobotRequest()
        request.joint_trajectory = goal_joint_pos

        response = move_robot(request)
        return response
    
    # def move_gripper(self, goal_gripper_pos):
    #     gripper_pos = GripperCmd()
    #     current_gripper_pos = self.gripper_states.position[0]
    #     gradient = (goal_gripper_pos - current_gripper_pos) / 20
    #     while current_gripper_pos < goal_gripper_pos:
    #         current_gripper_pos += gradient
    #         gripper_pos.position = current_gripper_pos
    #         self.gripper_publisher.publish(gripper_pos)

    
    # def move_direction(self, dir, q_init, kn):
    #     cur_xpos = self.get_xpos()
    #     qpos = xpos_to_qpos(xpos, kn, q_init)
    #     return self.move_step(qpos)
    
    
def make_env(camera_names):
    env = AlohaEnv(camera_names)
    return env



