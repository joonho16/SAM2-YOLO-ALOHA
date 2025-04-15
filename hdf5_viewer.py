import h5py
import numpy as np

import cv2

from constants import TASK_CONFIGS
        

def visualize_hdf5_images(hdf5_path, cameras=['camera1', 'camera2', 'camera3'], rgb2bgr=False):
    with h5py.File(hdf5_path, 'r') as f:
        images = []
        
        for im_name in cameras:
            images.append(f[f"observations/images/{im_name}"])

        episode_len = len(images[0])
        for i in range(episode_len):
            cur_img = []
            for (index, cam) in enumerate(cameras):
                if rgb2bgr:
                    correct_images = cv2.cvtColor(images[index][i], cv2.COLOR_RGB2BGR)
                else:
                    correct_images = images[index][i]
                cur_img.append(correct_images)
            
            max_height = 480
            max_width = 640
            resized_images = [cv2.resize(img, (max_width, max_height)) for img in cur_img]

            # 이미지를 가로로 나열
            combined_image = cv2.hconcat(resized_images)

            # 단일 창에 표시
            cv2.imshow("Combined Image", combined_image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                quit()


if __name__ == "__main__":
    # 기본값 설정
    dir = "./datasets"
    task_name = "pick_tomato"
    dataset_dir = TASK_CONFIGS[task_name]['dataset_dir']
    camera_names = TASK_CONFIGS[task_name]['camera_names']
    folder = "original"
    episode = "6"
    rgb2bgr = True

    # HDF5 파일 경로
    hdf5_path = f"{dataset_dir}/{folder}/episode_{episode}.hdf5"

    # 함수 호출
    while(1):
        visualize_hdf5_images(hdf5_path, camera_names)
    # for i in range(350):
    #     # HDF5 파일 경로
    #     hdf5_path = f"{dir}/{work}/original/episode_{str(i)}.hdf5"

    #     # 함수 호출
    #     visualize_hdf5_images(hdf5_path)