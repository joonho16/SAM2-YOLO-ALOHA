import kinematics
import torch
from curobo.types.base import TensorDeviceType
import h5py
import numpy as np
from constants import TASK_CONFIGS
from curobo.types.math import Pose

from utils import qpos_to_xpos

import cv2

# def hdf5_joints_to_tasks(hdf5_path, kin_config):
#     with h5py.File(hdf5_path, 'r+') as f:
        
#         if "observations/xpos" in f:
#             del f["observations/xpos"]  # 기존 데이터가 있으면 삭제 후 재작성
#         xpos_dataset = f.create_dataset("observations/xpos", (0, ), maxshape=(None, ), dtype='float32')

#         for qpos in f["observations/qpos"]:
#             tensor_qpos = torch.tensor(qpos, **(kin_config['tensor_args'].as_torch_dict()))
#             ee_pos = tensor_qpos[-1:]
#             tensor_qpos = tensor_qpos[:-1]
#             tensor_xpos = kinematics.forward_kinematics(tensor_qpos)
#             tensor_xpos = torch.cat((tensor_xpos, ee_pos), dim=0)
            
#             # HDF5에 데이터 저장 (확장 가능하게 설정)
#             xpos_dataset.resize((xpos_dataset.shape[0] + 1, tensor_xpos.shape[0]))
#             xpos_dataset[-1] = tensor_xpos.numpy()
#             print(xpos_dataset)


def hdf5_joints_to_tasks(hdf5_path, kin_config, home_pose):
    # Open file in read+write mode
    with h5py.File(hdf5_path, 'r+') as f:
        kn = kinematics.Kinematics(kin_config)
        # Read all qpos entries from the dataset
        qpos_data = f["observations/qpos"]
        action_data = f["/action"]
        
        num_samples = len(qpos_data)

        if "xpos" in f["observations"]:
            del f["observations"]["xpos"]
            
        if "xvel" in f["observations"]:
            del f["observations"]["xvel"]
            
        if "xaction" in f:
            del f["xaction"]
            
        if "xvel_action" in f:
            del f["xvel_action"]
            
        # if "qpos" in f["observations"]:
        #     del f["observations"]["qpos"]
            
        # if "action" in f:
        #     del f["action"]
        
        # Create the dataset for xpos with the shape [num_samples, xpos_dim]
        dset_xpos = f["observations"].create_dataset(
            "xpos",
            shape=(num_samples, 8),
            dtype=np.float32
        )
        dset_xvel = f["observations"].create_dataset(
            "xvel",
            shape=(num_samples, 8),
            dtype=np.float32
        )
        dset_xaction = f.create_dataset(
            "xaction",
            shape=(num_samples, 8),
            dtype=np.float32
        )
        dset_xvel_action = f.create_dataset(
            "xvel_action",
            shape=(num_samples, 8),
            dtype=np.float32
        )
        
        # dset_qpos = f["observations"].create_dataset(
        #     "qpos",
        #     shape=(num_samples, 7),
        #     dtype=np.float32
        # )
        # dset_action = f.create_dataset(
        #     "action",
        #     shape=(num_samples, 7),
        #     dtype=np.float32
        # )
        
        
        for i, qpos in enumerate(qpos_data):
            # ee_pos = torch.tensor(qpos[-1:], **(kin_config['tensor_args'].as_torch_dict()))
            # qpos_only = qpos[:-1]

            # xpos = kn.forward_kinematics(qpos_only)
            # xpos = torch.cat((xpos, ee_pos), dim=0)
            xpos = qpos_to_xpos(qpos, kn)
            dset_xpos[i] = xpos
            
            if i == 0:
                last_xpos = xpos
                
            xvel = xpos - last_xpos
            dset_xvel[i] = xvel
            
            last_xpos = xpos

            print(f"xpos: {dset_xpos[i]}")
            print(f"xvel: {dset_xvel[i]}")
        
        
            
            # # Optionally print or log
            # print(tensor_xpos)
        for i, action in enumerate(action_data):
            # ee_action = torch.tensor(action[-1:], **(kin_config['tensor_args'].as_torch_dict()))
            # qaction_only = action[:-1]

            # xaction = kn.forward_kinematics(qaction_only)
            # xaction = torch.cat((xaction, ee_action), dim=0)
            
            xaction = qpos_to_xpos(action, kn)
            dset_xaction[i] = xaction
            
            if i == 0:
                last_xaction = xaction
                
            xvel_action = xaction - last_xaction
            dset_xvel_action[i] = xvel_action
            
            # if i > 0:
            #     xvel_action = xaction - last_xaction
            #     dset_xvel_action[i-1] = xvel_action
            # if i == len(action_data) - 1:
            #     dset_xvel_action[i] = xvel_action
            
            last_xaction = xaction

            # Move to CPU NumPy array and store into the dataset

            print(f"xaction: {dset_xaction[i]}")
            print(f"xvel_action: {dset_xvel_action[i]}")
            
            # # Optionally print or log
            # print(tensor_xpos)

if __name__ == "__main__":
    tensor_args = TensorDeviceType()
    kin_config = {
        'robot_file': 'ur5e.yml',
        'world_file': "collision_base.yml",
        'rotation_threshold': 0.1,
        'position_threshold': 0.01,
        'num_seeds': 1,
        'self_collision_check': True,
        'self_collision_opt': False,
        'tensor_args': TensorDeviceType(),
        'use_cuda_graph': True
    }
    kn = kinematics.Kinematics(kin_config)
    
    # q_init = [-1.2082498, -1.406672, -0.91223973, -1.2803937, 1.3356875, -1.2104124]
    # xpos = [    0.0,     0.0,      0.1 ,    0.1 ,   -0.1,    -0.1  ,  -0.55503]
    # qpos = [-0.84704715,  -1.2372335,  -1.7173927, -0.43584868,  1.1489661,  -1.42844]
    
    # # result = kn.inverse_kinematics(xpos, q_init)
    # result = kn.forward_kinematics(qpos)
    
    # print(result)
    
    dir = "./datasets"
    task_name = "grasp_cable"
    dataset_dir = TASK_CONFIGS[task_name]['dataset_dir']
    home_pose = TASK_CONFIGS[task_name]['home_pose']
    folder = "original"
    # episode = "0"
    
    for i in range(27):

        # HDF5 파일 경로
        hdf5_path = f"{dataset_dir}/{folder}/episode_{i}.hdf5"

        hdf5_joints_to_tasks(hdf5_path, kin_config, home_pose)
        print(i)