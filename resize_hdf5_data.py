import h5py
import numpy
from utils import resize_image

dir = "./datasets"
work = "cap_a_bottle_with_tactile_old"
episode_len = 100

for episode in range(100):
    with h5py.File(f"{dir}/{work}/episode_{episode}.hdf5", 'r') as f:
        data_dict = {
            '/observations/qpos': f[f'/observations/qpos'],
            '/observations/qvel': f[f'/observations/qvel'],
            '/observations/effort': f[f'/observations/effort'],
            '/action': f[f'action'],
        }
        for cam_name in ['cam_1', 'cam_2', 'cam_3', 'digit']:
            data_dict[f'/observations/images/{cam_name}'] = []
            for frame in range(episode_len):
                data_dict[f'/observations/images/{cam_name}'].append(resize_image(f[f'/observations/images/{cam_name}'][frame]))

        dataset_path = f"{dir}/{work}_resized/episode_{episode}"
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in ['cam_1', 'cam_2', 'cam_3', 'digit']:
                _ = image.create_dataset(cam_name, (episode_len, 90, 160, 3), dtype='uint8',
                                        chunks=(1, 90, 160, 3), )

            _ = obs.create_dataset('qpos', (episode_len, 5))
            _ = obs.create_dataset('qvel', (episode_len, 5))
            _ = obs.create_dataset('effort', (episode_len, 5))
            _ = root.create_dataset('action', (episode_len, 5))
        
            for name, array in data_dict.items():
                root[name][...] = array
    print(f"{episode + 1}/100 completed")

    
