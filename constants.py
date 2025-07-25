DATA_DIR = './datasets'
TASK_CONFIGS = {
    'grasp_ball':{
        'dataset_dir': DATA_DIR + '/grasp_ball',
        'num_episodes': 1,
        'episode_len': 50,
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

DEFAULT_CAMERA_NAMES = ['camera/camera']

TOPIC_NAME = {
    'br_hand': {
        # 'joint_state': 'joint_state',
        # 'master_joint_state': 'master_joint_state',
        # 'master_gripper_state': 'master_joint_state',
    },
}

JOINT_NAMES = {
    'br_hand': ['thumb/FE', 'thumb/ABD', 'index/FE', 'middle/FE', 'ring/FE']
}

TOOL_NAMES = {
    # 'om': ['gripper'],
    # 'ur5': ['gripper']
}

JOINT_LIMIT = {
    'br_hand': {
        'thumb/FE': {
            'min': -4.35,
            'max': 0.41,
        },
        'thumb/ABD': {
            'min': 3.85,
            'max': 1.8,
        },
        'index/FE': {
            'min': -0.25,
            'max': 6.27,
        },
        'middle/FE': {
            'min': -4.83,
            'max': 1.9,
        },
        'ring/FE': {
            'min': -5.56,
            'max': 0.9,
        },
    },
}

DT = 0.1