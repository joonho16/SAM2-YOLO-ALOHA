DATA_DIR = './datasets'
TASK_CONFIGS = {
    'grasp_ball':{
        'dataset_dir': DATA_DIR + '/grasp_cable_yaskawa',
        'num_episodes': 36,
        'episode_len': 120,
        'camera_names': ['camera/camera'],
        'camera_config': {
            'camera/camera': {
                'resize': {
                    'size': (200, 150)
                },
            },
            
        },
        #'home_pose': [[0,0,0,0,0,-1.57, 0.087]],
        #'end_pose': [[0,0,0,0,0,-1.57, 0.087]],
        #'pose_sleep': 0
    }
}

TOPIC_NAME = {
    'br_hand': {
        # 'joint_state': 'joint_state',
        # 'master_joint_state': 'master_joint_state',
        # 'master_gripper_state': 'master_joint_state',
    },
}

JOINT_NAMES = {
    # 'br_hand': ['joint1', 'joint2', 'joint3', 'joint4']
}

TOOL_NAMES = {
    # 'om': ['gripper'],
    # 'ur5': ['gripper']
}

JOINT_LIMIT = {
    # 'br_hand': {
    #     'joint1': {
    #         'min': -3.142,
    #         'max': 3.142,
    #     },
    #     'joint2': {
    #         'min': -2.050,
    #         'max': 1.571,
    #     },
    #     'joint3': {
    #         'min': -1.571,
    #         'max': 1.530,
    #     },
    #     'joint4': {
    #         'min': -1.800,
    #         'max': 2.000,
    #     },
    #     'gripper': {
    #         'min': -0.01,
    #         'max': 0.01,
    #     },
    # },
}

DT = 0.1