DATA_DIR = './datasets'
TASK_CONFIGS = {
    'push_yellow_sponge':{
        'dataset_dir': DATA_DIR + '/push_yellow_sponge',
        'num_episodes': 50,
        'episode_len': 200,
        'camera_names': ['cam_1', 'cam_2', 'cam_3'],
    },
    'put_sponge_in_basket':{
        'dataset_dir': DATA_DIR + '/put_sponge_in_basket',
        'num_episodes': 100,
        'episode_len': 200,
        'camera_names': ['cam_1', 'cam_2', 'cam_3'],
    },
    'high_five':{
        'dataset_dir': DATA_DIR + '/high_five',
        'num_episodes': 30,
        'episode_len': 80,
        'camera_names': ['cam_1', 'cam_2', 'cam_3'],
    },
    'cap_a_bottle_with_tactile':{
        'dataset_dir': DATA_DIR + '/cap_a_bottle_with_tactile',
        'num_episodes': 130,
        'episode_len': 100,
        'camera_names': ['cam_1', 'cam_2', 'cam_3', 'digit'],
    },
    'cap_a_bottle':{
        'dataset_dir': DATA_DIR + '/cap_a_bottle_with_tactile',
        'num_episodes': 130,
        'episode_len': 100,
        'camera_names': ['cam_1', 'cam_2', 'cam_3'],
    },
    'cap_a_bottle_with_tactile_random_position':{
        'dataset_dir': DATA_DIR + '/cap_a_bottle_with_tactile_random_position',
        'num_episodes': 120,
        'episode_len': 100,
        'camera_names': ['cam_1', 'cam_2', 'cam_3', 'digit'],
    },
    'grasp_cable':{
        'dataset_dir': DATA_DIR + '/grasp_cable',
        'num_episodes': 258,
        'episode_len': 150,
        'camera_names': ['camera1', 'camera2'],
        'camera_config': {
            'camera1': {
                'zoom': {
                    'point': [300, 80],
                    'size': (160, 120)
                },
                # 'masked_yolo': {
                #     'model_path': 'yolo/runs/detect/train21/weights/best.pt',
                #     'classes': {
                #         'cable': {
                #             'id': 0,
                #             'is_fixed_mask': False,
                #             'show_id': 0,
                #             'keep_last_box': True
                #         },
                #         'gripper': {
                #             'id': 1,
                #             'is_fixed_mask': False,
                #             'show_id': 0,
                #             'keep_last_box': True,
                #         }
                #     }
                # },
            },
            'camera2': {
                'resize': {
                    'ratio': 4,
                    'size': (160, 120)
                },
                # 'masked_yolo': {
                #     'model_path': 'yolo/runs/detect/train21/weights/best.pt',
                #     'classes': {
                #         'cable': {
                #             'id': 0,
                #             'is_fixed_mask': False,
                #             'show_id': 0,
                #             'keep_last_box': True
                #         },
                #         'gripper': {
                #             'id': 1,
                #             'is_fixed_mask': False,
                #             'show_id': -1,
                #             'keep_last_box': True,
                #         }
                #     }
                # },
            }
        },
        'home_pose': [[-1.954, -1.108, -1.117, 0.158, 1.472, -1.511, 0.087]],
        # 'end_pose': [[-1.023, -2.528, -0.955, 0.314, 2.057, -1.661, 0]]
        'end_pose': [[-1.954, -1.108, -1.117, 0.158, 1.472, -1.511, 0.087]],
        'pose_sleep': 6
    },

    'pick_tomato':{
        'dataset_dir': DATA_DIR + '/pick_tomato_new',
        'num_episodes': 45,
        'episode_len': 150,
        'camera_names': ['camera1', 'camera2'],
        'camera_config': {
            'camera1': {
                'resize': {
                    'ratio': 1 / 4,
                    'size': (320, 240)
                },
                'masked_yolo': {
                    'model_path': 'yolo/runs/detect/train2/weights/best.pt',
                    'classes': {
                        'tomato': {
                            'id': 0,
                            'is_fixed_mask': True,
                            'show_id': 0,
                            'keep_last_box': False
                        },
                        'gripper': {
                            'id': 1,
                            'is_fixed_mask': False,
                            'show_id': -1,
                            'keep_last_box': True,
                        }
                    }
                },
                'size': (120, 160)
            },
            'camera2': {
                'resize': {
                    'ratio': 4,
                    'size': (160, 120)
                },
                'size': (120, 160)
            }
        },
        'home_pose': [[-2.29, -1.95, -1.66, -0.46, 1.71, -1.62, 0.087]],
        'end_pose': []
    },
}

DEFAULT_CAMERA_NAMES = ['cam_1', 'cam_2', 'cam_3']

TOPIC_NAME = {
    'om': {
        'joint_state': 'joint_state',
        'master_joint_state': 'master_joint_state',
        'master_gripper_state': 'master_joint_state',
    },
    'ur5': {
        'joint_state': 'ur5e/joint_states',
        'master_joint_state': 'ur5e/ur5e_scaled_pos_joint_traj_controller/command',
        'master_gripper_state': 'gripper/cmd',
    }
}

JOINT_NAMES = {
    'om': ['joint1', 'joint2', 'joint3', 'joint4'],
    'ur5': ["ur5e_elbow_joint", "ur5e_shoulder_lift_joint", "ur5e_shoulder_pan_joint", "ur5e_wrist_1_joint", "ur5e_wrist_2_joint", "ur5e_wrist_3_joint"]
}

TOOL_NAMES = {
    'om': ['gripper'],
    'ur5': ['gripper']
}

JOINT_LIMIT = {
    'om': {
        'joint1': {
            'min': -3.142,
            'max': 3.142,
        },
        'joint2': {
            'min': -2.050,
            'max': 1.571,
        },
        'joint3': {
            'min': -1.571,
            'max': 1.530,
        },
        'joint4': {
            'min': -1.800,
            'max': 2.000,
        },
        'gripper': {
            'min': -0.01,
            'max': 0.01,
        },
    },
    'ur5': {
        'ur5e_elbow_joint': {
            'min': -3.142,
            'max': 3.142,
        },
        'ur5e_shoulder_lift_joint': {
            'min': -3.142,
            'max': 3.142,
        },
        'ur5e_shoulder_pan_joint': {
            'min': -3.142,
            'max': 3.142,
        },
        'ur5e_wrist_1_joint': {
            'min': -3.142,
            'max': 3.142,
        },
        'ur5e_wrist_2_joint': {
            'min': -3.142,
            'max': 3.142,
        },
        'ur5e_wrist_3_joint': {
            'min': -3.142,
            'max': 3.142,
        },
        'gripper': {
            'min': -3.142,
            'max': 3.142,
        },
    }
}

YOLO_CONFIG = {
    'pick_tomato': {
        'class_names': ['tomato', 'gripper'],
        'raw_data_dir': 'yolo/raw_data',
        'data_dir': 'yolo/data',
        'tmp_data_dir': 'yolo/tmp_data',
        'yaml_dir': 'yolo',
        'yolo_path': 'yolo/yolo11n.pt',
        'checkpoint': 'sam2/checkpoints/sam2.1_hiera_large.pt',
        'sam2_config_dir': 'sam2/sam2/configs/sam2.1',
        'model_cfg_yaml': 'sam2.1_hiera_l.yaml',
        'epochs': 100
    },
        'grasp_cable': {
        'class_names': ['cable_head', 'gripper'],
        'raw_data_dir': 'yolo/raw_data/grasp_cable',
        'data_dir': 'yolo/data',
        'tmp_data_dir': 'yolo/tmp_data',
        'yaml_dir': 'yolo',
        'yolo_path': 'yolo/yolo11n.pt',
        'checkpoint': 'sam2/checkpoints/sam2.1_hiera_large.pt',
        'sam2_config_dir': 'sam2/sam2/configs/sam2.1',
        'model_cfg_yaml': 'sam2.1_hiera_l.yaml',
        'epochs': 100
    },
    'hand': {
        'class_names': ['hand'],
        'raw_data_dir': 'yolo/raw_data',
        'data_dir': 'yolo/data',
        'tmp_data_dir': 'yolo/tmp_data',
        'yaml_dir': 'yolo',
        'yolo_path': 'yolo/yolo11n.pt',
        'checkpoint': 'sam2/checkpoints/sam2.1_hiera_large.pt',
        'sam2_config_dir': 'sam2/sam2/configs/sam2.1',
        'model_cfg_yaml': 'sam2.1_hiera_l.yaml',
        'epochs': 10
    }
}

DT = 0.1