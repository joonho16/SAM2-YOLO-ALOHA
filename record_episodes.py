from env import AlohaEnv
import time
import argparse
from tqdm import tqdm
import os
import h5py
from constants import DT, TASK_CONFIGS, JOINT_NAMES
import rospy
import numpy as np
import cv2

from ultralytics import YOLO
from apply_yolo import mask_outside_boxes

from utils import zoom_image

def capture_one_episode(env, max_timesteps, dataset_dir, camera_names, camera_config, dataset_name, overwrite=True):

    env.go_home_pose()
    
    print(f'Dataset name: {dataset_name}')

    joint_len = len(JOINT_NAMES['ur5']) + 1

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    ts = env.reset()
    timesteps = [ts]
    actions = []
    for t in tqdm(range(max_timesteps)):
        action = env.get_action()
        actions.append(action)
        ts = env.record_step()
        timesteps.append(ts)

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    timesteps.pop(len(timesteps) - 1)

    is_first_img = True
    while timesteps:

        ts = timesteps.pop(0)
        action = actions.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)

        for cam_name in camera_names:
            image = ts.observation['images'][cam_name]
            if cam_name in camera_config:
                if 'zoom' in camera_config[cam_name]:
                    size = camera_config[cam_name]['zoom']['size']
                    point = camera_config[cam_name]['zoom']['point']
                    image = zoom_image(image, point, size)
                if 'resize' in camera_config[cam_name]:
                    rate = camera_config[cam_name]['resize']['size']
                    image = cv2.resize(image, size)
                    

            if ts.observation['images'][cam_name] is not None:
                data_dict[f'/observations/images/{cam_name}'].append(image)
            else:
                print("error")

    
    # HDF5
    t0 = time.time()
    image_size = (120, 160)
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, image_size[0], image_size[1], 3), dtype='uint8',
                                    chunks=(1, image_size[0], image_size[1], 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        _ = obs.create_dataset('qpos', (max_timesteps, joint_len))
        _ = obs.create_dataset('qvel', (max_timesteps, joint_len))
        _ = obs.create_dataset('effort', (max_timesteps, joint_len))
        _ = root.create_dataset('action', (max_timesteps, joint_len))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs')

    return True

def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def main(args):
    task_config = TASK_CONFIGS[args['task']]
    dataset_dir = f"{task_config['dataset_dir']}/original" 
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']
    camera_config = task_config['camera_config']

    rospy.init_node("record_episode_node", anonymous=False)
    _env = AlohaEnv(camera_names)

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    dataset_name = f'episode_{episode_idx}'
    capture_one_episode(_env, max_timesteps, dataset_dir, camera_names, camera_config, dataset_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    main(vars(parser.parse_args()))