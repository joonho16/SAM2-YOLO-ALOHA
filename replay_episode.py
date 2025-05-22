import h5py
import numpy as np

from env import AlohaEnv

import time
from constants import TASK_CONFIGS

import rospy

import cv2

from tqdm import tqdm
from utils import sample_box_pose, sample_insertion_pose, qpos_to_xpos, xpos_to_qpos # robot functions


def replay_episode(hdf5_path, task_config, task_space, vel_control, kn=None):
    home_pose = task_config['home_pose']
    end_pose = task_config['end_pose']
    pose_sleep = task_config['pose_sleep']
    camera_names = task_config['camera_names']
    episode_len = task_config['episode_len']

    images = []

        
    with h5py.File(hdf5_path, 'r') as f:
        actions = f[f"action"][:]
        xactions = f[f"xaction"][:]
        xvel_actions = f[f"xvel_action"][:]
        xpos_data = f["observations/xpos"][:]
        qpos_data = f["observations/qpos"][:]
        xvel_data = f["observations/xvel"][:]

        for im_name in camera_names:
            images.append(f[f"observations/images/{im_name}"])

        
        rospy.init_node("replay_episode_node", anonymous=False)
        
        env = AlohaEnv(camera_names, kn=kn, robot_name='yaskawa')

        env.move_joint(actions[0])

        time.sleep(pose_sleep)

        for i in tqdm(range(len(actions))):
            qaction = actions[i]
            xaction = xactions[i]
            xvel_action = xvel_actions[i]
            
            xpos = xpos_data[i]
            qpos = qpos_data[i]
            xvel = xvel_data[i]
            if task_space:
                if vel_control:
                    cur_xpos = env.get_xpos()
                    print('cur_xpos: ', cur_xpos)
                    target_xpos = cur_xpos - xvel_action
                    print('target_xpos: ', target_xpos)
                    action = xpos_to_qpos(target_xpos, kn, qaction[:-1])
                else:
                    action = xpos_to_qpos(xaction, kn, qaction[:-1])
            else:
                action = qaction
                # print(kn.forward_kinematics(action))

            # print(f"xpos:{xpos}")
            # print(f"xact:{xaction}")
            print(f"qpos:{qpos}")
            print(f"action:{action}")
            # print(f"xvel_action:{xvel_action}")
            # print(f"xvel:{xvel}")
            env.move_step(action)
            
            cur_img = []
            for (index, cam) in enumerate(camera_names):
                cur_img.append(images[index][i])
            

            time.sleep(0.2)
            
            max_height = 480
            max_width = 640
            resized_images = [cv2.resize(img, (max_width, max_height)) for img in cur_img]

            # 이미지를 가로로 나열
            combined_image = cv2.hconcat(resized_images)

            # 단일 창에 표시
            cv2.imshow("Combined Image", combined_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                quit()

        # for pos in end_pose:
        #     env.move_joint(pos)
        #     time.sleep(pose_sleep)
            

if __name__ == "__main__":
    # 기본값 설정
    dir = "./datasets"
    # work = "pick_tomato"
    work = "grasp_cable_yaskawa"
    episode = "1"
    task_config = TASK_CONFIGS[work]

    task_space = False
    vel_control = True
    
    kn = None
    if task_space:

        import kinematics
        from curobo.types.base import TensorDeviceType

        kin_config = {
            'robot_file': 'ur5e.yml',
            'world_file': "collision_base.yml",
            'rotation_threshold': 0.1,
            'position_threshold': 0.01,
            'num_seeds': 500,
            'self_collision_check': True,
            'self_collision_opt': False,
            'tensor_args': TensorDeviceType(),
            'use_cuda_graph': True
        }
        kn = kinematics.Kinematics(kin_config)


    # HDF5 파일 경로
    hdf5_path = f"{task_config['dataset_dir']}/original/episode_{episode}.hdf5"
    

    # 함수 호출
    replay_episode(hdf5_path, task_config, task_space, vel_control, kn)